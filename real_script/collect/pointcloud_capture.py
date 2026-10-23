import cv2
import os
import sys
import json
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera, Session, cam_code, camera_metadata
from camera import open_camera

# --- 設定読み込み ---
_parser = build_parser()
_parser.add_argument('--frames', type=int, default=None, metavar='N',
                     help='autoモードのフレーム数（config.yaml の値を上書き）')
_parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                     help='auto: N フレーム自動取得 | manual: [s]キーで1枚ずつ取得')
_parser.add_argument('--min-depth', type=float, default=None, dest='min_depth', metavar='F',
                     help='深度フィルタの最小距離 (m)（config.yaml の値を上書き）')
_parser.add_argument('--max-depth', type=float, default=None, dest='max_depth', metavar='F',
                     help='深度フィルタの最大距離 (m)（config.yaml の値を上書き）')
_parser.add_argument('--no-filter', action='store_true',
                     help='深度フィルタを無効にして生データをそのまま取得')
_args = _parser.parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']
capture_frames = _args.frames if _args.frames is not None else _cfg['pointcloud']['capture_frames']
mode = _args.mode

min_d = _args.min_depth if _args.min_depth is not None else _cfg['pointcloud']['min_depth']
max_d = _args.max_depth if _args.max_depth is not None else _cfg['pointcloud']['max_depth']
use_filter = not _args.no_filter

# --- カメラ検出 ---
try:
    _cam = detect_camera(_args.serial)
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")

# --- 出力ディレクトリ ---
# color/depth/ply は点群処理でまとめて扱うためセッション直下にフラットに置く
# （point_merge.py が session_dir 直下を走査する）
session  = Session(_cfg['output']['pointcloud_dir'], cam_code(_cam['model']), tag=_args.tag)
save_dir = str(session.dir)

print(f"保存先: {save_dir}")
mode_label = f"auto ({capture_frames} frames)" if mode == 'auto' else "manual"
print(f"モード: {mode_label}")
print(f"解像度: {W}x{H} @ {FPS}fps")
if use_filter:
    print(f"深度フィルタ: {min_d}m 〜 {max_d}m  (生データは --no-filter で取得可)")
else:
    print(f"深度フィルタ: 無効（生データ）")
print()

# --- カメラ セットアップ（color + depth のみ） ---

camera = open_camera(_cam, W, H, FPS)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    sys.exit(1)

# --- カメラ内部パラメータ保存 ---
with open(os.path.join(save_dir, 'intrinsics.json'), 'w') as f:
    json.dump(camera.intrinsics(), f, indent=2)

def save_frame(idx, fr):
    cv2.imwrite(session.path(idx, 'color', sub=False), fr.color)
    cv2.imwrite(session.path(idx, 'depth', ext='png', sub=False),
                fr.depth)                           # フィルタなし・16-bit PNG

    # PLY は深度フィルタの範囲内の点だけで生成（--no-filter 時は生のまま）
    camera.save_ply(fr, session.path(idx, 'pointcloud', ext='ply', sub=False),
                    *((min_d, max_d) if use_filter else (None, None)))


def write_metadata(actual_frames):
    session.write_metadata(
        camera=camera_metadata(_cam, W, H, FPS, camera),
        mode=mode,
        capture_frames=capture_frames if mode == 'auto' else None,
        actual_frames=actual_frames,
        depth_filter={
            'enabled':   use_filter,
            'min_depth': min_d if use_filter else None,
            'max_depth': max_d if use_filter else None,
        },
    )


frame_count = 0

try:
    if mode == 'auto':
        # プレビューループ（Enter で開始）
        print("[Enter] で取得開始  [q] で中止")
        while True:
            fr = camera.read()
            if fr is None:
                continue
            preview = fr.color.copy()
            cv2.putText(preview, "[Enter] Start  [q] Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('pointcloud_capture', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:   # Enter
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("中止しました。")
                sys.exit(0)

        # 自動取得ループ
        print("取得中...")
        while frame_count < capture_frames:
            fr = camera.read()
            if fr is None:
                continue
            frame_count += 1
            save_frame(frame_count, fr)
            print(f"\rsaved: {frame_count}/{capture_frames} frames", end="", flush=True)

        print()   # 改行

    else:  # manual
        print("[s] で1フレーム取得  [q] で終了")
        while True:
            fr = camera.read()
            if fr is None:
                continue

            preview = fr.color.copy()
            cv2.putText(preview, f"[s] Save ({frame_count} saved)  [q] Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('pointcloud_capture', preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                frame_count += 1
                save_frame(frame_count, fr)
                print(f"saved: {frame_count} frames")
            elif key == ord('q'):
                break

finally:
    camera.stop()
    cv2.destroyAllWindows()
    write_metadata(frame_count)
    print(f"完了: {frame_count} フレーム保存 → {save_dir}")
    gc.collect()
