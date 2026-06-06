import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc
import yaml
from pathlib import Path
from datetime import datetime

with open(Path(__file__).parent.parent / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

# --- 1. 保存ディレクトリの設定 ---
save_dir_base = os.path.expanduser(_cfg['output']['images_dir'])
os.makedirs(save_dir_base, exist_ok=True)

date_str = datetime.now().strftime('%Y-%m-%d')
date_dir = os.path.join(save_dir_base, date_str)
os.makedirs(date_dir, exist_ok=True)

print(f"画像を {date_dir} に保存します")
print("\nストリーミング準備中...")

# --- 2. RealSenseの初期化 ---
pipeline = rs.pipeline()
config = rs.config()
ctx = rs.context()

devices = ctx.query_devices()
if len(devices) == 0:
    print("No Intel Device connected")
    exit(0)

for dev in devices:
    print('Found device:', dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))

try:
    config.enable_stream(rs.stream.color,      W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth,      W, H, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8,  FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8,  FPS)
except RuntimeError as e:
    print(f"ストリームの設定に失敗しました: {e}")
    exit(0)

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
        ir_frame1   = aligned_frames.get_infrared_frame(1)
        ir_frame2   = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1   = np.asanyarray(ir_frame1.get_data())
        ir_image2   = np.asanyarray(ir_frame2.get_data())

        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        preview = np.vstack((
            np.hstack((ir_colormap1, ir_colormap2)),
            np.hstack((color_image,  depth_colormap))
        ))

        cv2.putText(preview, f"[s] Save  [q] Quit   Saved: {shot_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('RealSense', preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime('%H%M%S_%f')[:-2]

            path_color          = os.path.join(date_dir, "color");          os.makedirs(path_color, exist_ok=True)
            path_depth          = os.path.join(date_dir, "depth");          os.makedirs(path_depth, exist_ok=True)
            path_ir_left        = os.path.join(date_dir, "ir_left");        os.makedirs(path_ir_left, exist_ok=True)
            path_ir_right       = os.path.join(date_dir, "ir_right");       os.makedirs(path_ir_right, exist_ok=True)
            path_ir_left_color  = os.path.join(date_dir, "ir_left_color");  os.makedirs(path_ir_left_color, exist_ok=True)
            path_ir_right_color = os.path.join(date_dir, "ir_right_color"); os.makedirs(path_ir_right_color, exist_ok=True)

            cv2.imwrite(os.path.join(path_color,          f"{timestamp}_color.jpg"),           color_image)
            cv2.imwrite(os.path.join(path_depth,          f"{timestamp}_depth_colormap.jpg"),  depth_colormap)
            cv2.imwrite(os.path.join(path_ir_left,        f"{timestamp}_ir_left.jpg"),         ir_image1)
            cv2.imwrite(os.path.join(path_ir_right,       f"{timestamp}_ir_right.jpg"),        ir_image2)
            cv2.imwrite(os.path.join(path_ir_left_color,  f"{timestamp}_ir_left_color.jpg"),   ir_colormap1)
            cv2.imwrite(os.path.join(path_ir_right_color, f"{timestamp}_ir_right_color.jpg"),  ir_colormap2)

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
        del ir_frame1, ir_image1
        del ir_frame2, ir_image2
        del aligned_frames, frames
    except NameError:
        pass

    gc.collect()
