import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera, get_depth_alpha, make_depth_colormap

_args = build_parser().parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

# --- 1. 保存先のベースディレクトリ ---
save_dir_base = os.path.expanduser(_cfg['output']['images_dir'])
_now     = datetime.now()
date_key = _now.strftime('%Y_%m%d')   # 例: 2026_0624
date_str = _now.strftime('%Y-%m-%d')  # 例: 2026-06-24
time_str = _now.strftime('%H%M%S')    # 例: 101741
save_dir_dated = os.path.join(save_dir_base, date_key)
os.makedirs(save_dir_dated, exist_ok=True)

print("\nストリーミング準備中...")

# --- 2. RealSenseの初期化 ---
try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir      = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

existing_sessions = [d for d in os.scandir(save_dir_dated) if d.is_dir() and d.name.startswith('image')]
N = len(existing_sessions) + 1
base_path = os.path.join(save_dir_dated, f"image{N}_{date_str}_{time_str}_{_cam['model']}")
os.makedirs(base_path)
print(f"画像を {base_path} に保存します")

pipeline = rs.pipeline()
config = rs.config()

try:
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)
    if _has_ir:
        config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
        config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
except RuntimeError as e:
    print(f"ストリームの設定に失敗しました: {e}")
    exit(1)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


# --- 3. メインループ ---
try:
    print("\n[s] キー : 画像を1枚保存")
    print("[q] キー : 終了\n")

    shot_count = 0

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image    = np.asanyarray(color_frame.get_data())
        depth_image    = np.asanyarray(depth_frame.get_data())
        depth_colormap = make_depth_colormap(depth_image, _depth_alpha)

        if _has_ir:
            ir_frame1 = aligned_frames.get_infrared_frame(1)
            ir_frame2 = aligned_frames.get_infrared_frame(2)
            if not ir_frame1 or not ir_frame2:
                continue
            ir_image1    = np.asanyarray(ir_frame1.get_data())
            ir_image2    = np.asanyarray(ir_frame2.get_data())
            ir_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
            ir_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
            preview = np.vstack((
                np.hstack((ir_colormap1, ir_colormap2)),
                np.hstack((color_image,  depth_colormap))
            ))
        else:
            preview = np.hstack((color_image, depth_colormap))

        cv2.putText(preview, f"[s] Save  [q] Quit   Saved: {shot_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('RealSense', preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime('%H%M%S_%f')[:-2]

            path_color = os.path.join(base_path, "color"); os.makedirs(path_color, exist_ok=True)
            path_depth = os.path.join(base_path, "depth"); os.makedirs(path_depth, exist_ok=True)

            cv2.imwrite(os.path.join(path_color, f"{timestamp}_color.jpg"),          color_image)
            cv2.imwrite(os.path.join(path_depth, f"{timestamp}_depth_colormap.jpg"), depth_colormap)

            if _has_ir:
                path_ir_left        = os.path.join(base_path, "ir_left");        os.makedirs(path_ir_left, exist_ok=True)
                path_ir_right       = os.path.join(base_path, "ir_right");       os.makedirs(path_ir_right, exist_ok=True)
                path_ir_left_color  = os.path.join(base_path, "ir_left_color");  os.makedirs(path_ir_left_color, exist_ok=True)
                path_ir_right_color = os.path.join(base_path, "ir_right_color"); os.makedirs(path_ir_right_color, exist_ok=True)
                cv2.imwrite(os.path.join(path_ir_left,        f"{timestamp}_ir_left.jpg"),        ir_image1)
                cv2.imwrite(os.path.join(path_ir_right,       f"{timestamp}_ir_right.jpg"),       ir_image2)
                cv2.imwrite(os.path.join(path_ir_left_color,  f"{timestamp}_ir_left_color.jpg"),  ir_colormap1)
                cv2.imwrite(os.path.join(path_ir_right_color, f"{timestamp}_ir_right_color.jpg"), ir_colormap2)

            shot_count += 1
            print(f"[{shot_count}枚目保存] {timestamp}")

        elif key == ord('q'):
            print(f"\n終了します。合計 {shot_count} 枚保存しました。")
            break

finally:
    print("ストリーミングを停止し、リソースを解放します。")
    pipeline.stop()
    cv2.destroyAllWindows()

    try:
        del color_frame, color_image
        del depth_frame, depth_image
        del aligned_frames, frames
    except NameError:
        pass

    gc.collect()
