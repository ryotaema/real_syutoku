import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args

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

# --- 出力ディレクトリ ---
base_dir    = os.path.expanduser(_cfg['pointcloud']['output_dir'])
date_str    = datetime.now().strftime('%Y-%m-%d')
session_str = datetime.now().strftime('session_%H%M%S')
save_dir    = os.path.join(base_dir, date_str, session_str)
os.makedirs(save_dir, exist_ok=True)

print(f"保存先: {save_dir}")
mode_label = f"auto ({capture_frames} frames)" if mode == 'auto' else "manual"
print(f"モード: {mode_label}")
print(f"解像度: {W}x{H} @ {FPS}fps")
if use_filter:
    print(f"深度フィルタ: {min_d}m 〜 {max_d}m  (生データは --no-filter で取得可)")
else:
    print(f"深度フィルタ: 無効（生データ）")
print()

# --- RealSense セットアップ（color + depth のみ） ---
pipeline  = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
rs_config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)

profile = pipeline.start(rs_config)
align   = rs.align(rs.stream.color)
pc_obj  = rs.pointcloud()

# --- 深度フィルタ設定 ---
depth_filter = None
if use_filter:
    depth_filter = rs.threshold_filter()
    depth_filter.set_option(rs.option.min_distance, min_d)
    depth_filter.set_option(rs.option.max_distance, max_d)

# --- カメラ内部パラメータ保存 ---
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
intrinsics_data = {
    'width':  intr.width,
    'height': intr.height,
    'fx':     intr.fx,
    'fy':     intr.fy,
    'ppx':    intr.ppx,
    'ppy':    intr.ppy,
    'model':  str(intr.model),
    'coeffs': list(intr.coeffs),
}
with open(os.path.join(save_dir, 'intrinsics.json'), 'w') as f:
    json.dump(intrinsics_data, f, indent=2)

# --- セッションメタデータ初期化 ---
metadata = {
    'mode':           mode,
    'capture_frames': capture_frames if mode == 'auto' else None,
    'actual_frames':  0,
    'width':          W,
    'height':         H,
    'fps':            FPS,
    'depth_filter': {
        'enabled':   use_filter,
        'min_depth': min_d if use_filter else None,
        'max_depth': max_d if use_filter else None,
    },
    'timestamp': datetime.now().isoformat(),
}


def save_frame(idx, c_frame, d_frame):
    prefix = os.path.join(save_dir, f"{idx:04d}")

    color     = np.asanyarray(c_frame.get_data())   # RGB
    depth_raw = np.asanyarray(d_frame.get_data())   # uint16 生深度
    cv2.imwrite(prefix + '_color.jpg',  cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(prefix + '_depth.png',  depth_raw)  # フィルタなし・16-bit PNG

    # PLY はフィルタ適用後の深度で生成（--no-filter 時は生のまま）
    depth_for_pc = depth_filter.process(d_frame) if depth_filter else d_frame
    pc_obj.map_to(c_frame)
    pts = pc_obj.calculate(depth_for_pc)
    pts.export_to_ply(prefix + '_pointcloud.ply', c_frame)


def write_metadata(actual_frames):
    metadata['actual_frames'] = actual_frames
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


frame_count = 0

try:
    if mode == 'auto':
        # プレビューループ（Enter で開始）
        print("[Enter] で取得開始  [q] で中止")
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            if not c_frame:
                continue
            preview = cv2.cvtColor(np.asanyarray(c_frame.get_data()), cv2.COLOR_RGB2BGR)
            cv2.putText(preview, "[Enter] Start  [q] Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('pointcloud_capture', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:   # Enter
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("中止しました。")
                pipeline.stop()
                gc.collect()
                sys.exit(0)

        # 自動取得ループ
        print("取得中...")
        while frame_count < capture_frames:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue
            save_frame(frame_count, c_frame, d_frame)
            frame_count += 1
            print(f"\rsaved: {frame_count}/{capture_frames} frames", end="", flush=True)

        print()   # 改行

    else:  # manual
        print("[s] で1フレーム取得  [q] で終了")
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            preview = cv2.cvtColor(np.asanyarray(c_frame.get_data()), cv2.COLOR_RGB2BGR)
            cv2.putText(preview, f"[s] Save ({frame_count} saved)  [q] Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('pointcloud_capture', preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_frame(frame_count, c_frame, d_frame)
                frame_count += 1
                print(f"saved: {frame_count} frames")
            elif key == ord('q'):
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    write_metadata(frame_count)
    print(f"完了: {frame_count} フレーム保存 → {save_dir}")
    gc.collect()
