import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (load_config, build_parser, apply_args, detect_camera, get_depth_alpha,
                   make_depth_colormap, Session, cam_code)

WARMUP_SECS = 2.0  # AE安定待ち（フレーム数でなく秒数で管理）


def _build_parser():
    parser = build_parser(include_conf=True)
    parser.add_argument('--interval', type=int, default=300,
                        help='撮影間隔（秒）デフォルト: 300（5分）')
    parser.add_argument('--duration', type=float, default=12.0,
                        help='撮影継続時間（時間）デフォルト: 12.0')
    parser.add_argument('--detect', action='store_true',
                        help='起動時にGUIでYOLOモデルを選択しBBOX付き画像も保存する')
    parser.add_argument('--relative-depth', action='store_true',
                        help='深度カラーマップをフレーム内の相対値で正規化する（学習用途には非推奨）')
    return parser


_args = _build_parser().parse_args()
_cfg  = apply_args(load_config(), _args)

W, H, FPS   = _cfg['camera']['width'], _cfg['camera']['height'], _cfg['camera']['fps']
CONF        = _cfg['model']['confidence_threshold']
INTERVAL    = _args.interval
if INTERVAL <= 0:
    print("--interval は1以上を指定してください。")
    sys.exit(1)
TOTAL_SEC   = int(_args.duration * 3600)
TOTAL_SHOTS = TOTAL_SEC // INTERVAL

# ── カメラ検出 ───────────────────────────────────────────────────────────────
try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    sys.exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir      = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# ── --detect: GUIでモデルを選択してYOLOを読み込む ─────────────────────────────
model    = None
log_path = None

if _args.detect:
    _model_path = ''
    try:
        import tkinter as tk
        from tkinter import filedialog
        _tk = tk.Tk()
        _tk.withdraw()
        _model_path = filedialog.askopenfilename(
            title='YOLOモデルを選択（.pt）',
            initialdir=str(Path(__file__).parent.parent / 'model'),
            filetypes=[('YOLO model', '*.pt'), ('All files', '*.*')]
        )
        _tk.destroy()
    except ImportError:
        print("tkinter が必要です: sudo apt install python3-tk")
        sys.exit(1)
    except Exception as e:
        print(f"GUIの起動に失敗しました: {e}")
        sys.exit(1)

    if not _model_path:
        print("モデルが選択されませんでした。--detect なしで起動します。")
        _args.detect = False
    else:
        from ultralytics import YOLO
        model = YOLO(_model_path)
        print(f"モデル読み込み: {_model_path}")

# ── 出力先 ──────────────────────────────────────────────────────────────────
_images_base    = Path(os.path.expanduser(_cfg['output']['images_dir']))
_timelapse_base = _images_base.parent / 'timelapse_data'

_mods = ['color', 'depth', 'depth_colormap']
if _has_ir:
    _mods += ['ir_left', 'ir_right']
if _args.detect:
    _mods.append('annotated')

session     = Session(_timelapse_base, cam_code(_cam['model']), tag=_args.tag, subdirs=_mods)
session_dir = session.dir
if _args.detect:
    log_path = session_dir / 'detection_log.csv'

# ── RealSense 初期化 ─────────────────────────────────────────────────────────
pipeline = rs.pipeline()
rs_cfg   = rs.config()
rs_cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
rs_cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)
if _has_ir:
    rs_cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    rs_cfg.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
pipeline.start(rs_cfg)

print(f"オートエクスポージャ安定待ち（{WARMUP_SECS}秒）...")
_warmup_end = time.time() + WARMUP_SECS
while time.time() < _warmup_end:
    pipeline.wait_for_frames()

align = rs.align(rs.stream.color)

# ── CSV ヘッダー（--detect 時のみ）───────────────────────────────────────────
if log_path:
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['timestamp', 'elapsed_min', 'num_detections',
             'avg_conf', 'max_conf', 'classes'])

print(f"\n保存先    : {session_dir}")
print(f"撮影間隔  : {INTERVAL}秒（{INTERVAL/60:.1f}分）")
print(f"継続時間  : {_args.duration}時間（予定 {TOTAL_SHOTS} 枚）")
print(f"YOLO検出  : {'有効（信頼度閾値 ' + str(CONF) + '）' if _args.detect else '無効'}\n")


def _capture(shot_idx: int, start: float) -> bool:
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        print(f"[警告] フレーム欠落（shot {shot_idx}）- スキップ")
        return False

    color_img = np.asanyarray(cf.get_data())
    depth_img = np.asanyarray(df.get_data())
    # --relative-depth または D405（_depth_alpha=None）は相対正規化
    depth_vis = make_depth_colormap(depth_img, None if _args.relative_depth else _depth_alpha)

    writes = {
        session.path(shot_idx, 'color'):                  color_img,
        session.path(shot_idx, 'depth', ext='png'):       depth_img,
        session.path(shot_idx, 'depth_colormap'):         depth_vis,
    }

    if _has_ir:
        # IRはalign前のフレームから取得（元解像度・純粋なIR画像）
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)
        if ir1 and ir2:
            writes[session.path(shot_idx, 'ir_left')]  = np.asanyarray(ir1.get_data())
            writes[session.path(shot_idx, 'ir_right')] = np.asanyarray(ir2.get_data())

    write_ok = True
    for path, img in writes.items():
        if not cv2.imwrite(path, img):
            print(f"[警告] 書き込み失敗: {os.path.basename(path)}")
            write_ok = False

    elapsed_min = (time.time() - start) / 60
    now_str     = datetime.now().isoformat(timespec='seconds')
    log_line    = f"[{shot_idx:3d}/{TOTAL_SHOTS}] {now_str}  {elapsed_min:.1f}min"

    if _args.detect and model is not None:
        res     = model(color_img, conf=CONF, show=False, save=False, verbose=False)[0]
        boxes   = res.boxes
        n       = len(boxes)
        confs   = boxes.conf.cpu().numpy() if n > 0 else np.array([])
        avg_c   = float(confs.mean()) if n > 0 else 0.0
        max_c   = float(confs.max())  if n > 0 else 0.0
        classes = ','.join(res.names[int(c)] for c in boxes.cls.cpu().numpy()) if n > 0 else ''

        if not cv2.imwrite(session.path(shot_idx, 'annotated'), res.plot()):
            print(f"[警告] 書き込み失敗: {session.name(shot_idx, 'annotated')}")
            write_ok = False

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [now_str, f'{elapsed_min:.1f}', n,
                 f'{avg_c:.4f}', f'{max_c:.4f}', classes])

        log_line += f"  検出:{n:2d}個  avg_conf:{avg_c:.3f}"

    print(log_line)
    return write_ok


shot_count = 0
save_count = 0

try:
    start_time = time.time()
    next_time  = start_time

    while True:
        now = time.time()

        if now - start_time >= TOTAL_SEC:
            print(f"\n設定時間 {_args.duration}時間 経過。終了します。")
            break

        if now >= next_time:
            shot_count += 1
            try:
                if _capture(shot_count, start_time):
                    save_count += 1
            except RuntimeError as e:
                print(f"[警告] フレーム取得失敗（shot {shot_count}）: {e}")
            # ドリフト防止: 基準時刻からの掛け算で次回時刻を算出
            next_time = start_time + shot_count * INTERVAL
            if shot_count >= TOTAL_SHOTS:
                print(f"\n予定 {TOTAL_SHOTS} 枚完了。終了します。")
                break
        else:
            try:
                pipeline.wait_for_frames(timeout_ms=200)
            except RuntimeError:
                time.sleep(0.05)

finally:
    session.write_metadata(
        camera={'name': _cam['name'], 'model': _cam['model'], 'serial': _cam['serial'],
                'resolution': [W, H], 'fps': FPS},
        modalities=_mods,
        shot_count=save_count,
        timelapse={'interval_sec': INTERVAL, 'duration_hour': _args.duration,
                   'planned_shots': TOTAL_SHOTS, 'attempted': shot_count},
        detect={'enabled': bool(_args.detect), 'model': _model_path if _args.detect else None,
                'conf': CONF} if _args.detect else None,
    )
    pipeline.stop()
    print(f"\n完了。試行 {shot_count} 枚 / 保存成功 {save_count} 枚。ログ: {log_path or 'なし'}")
