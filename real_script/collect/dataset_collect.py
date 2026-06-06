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
i = 0
j = 1

save_dir_base = os.path.expanduser(_cfg['output']['images_dir'])
date_str = datetime.now().strftime('%Y-%m-%d')
save_dir_dated = os.path.join(save_dir_base, date_str)
os.makedirs(save_dir_dated, exist_ok=True)

file_ok = True
base_path = ""
while file_ok:
    base_path = os.path.join(save_dir_dated, f"image_{j}")
    try:
        os.makedirs(base_path)
        file_ok = False
    except FileExistsError:
        print(f"フォルダが存在しています: {base_path}")
        file_ok = True
    j += 1

path1 = os.path.join(base_path, "color/")
path2 = os.path.join(base_path, "depth/")
path3 = os.path.join(base_path, "ir_left/")
path4 = os.path.join(base_path, "ir_right/")
path5 = os.path.join(base_path, "ir_left_color/")
path6 = os.path.join(base_path, "ir_right_color/")

print(f"画像を {base_path} に保存します")

for p in [path1, path2, path3, path4, path5, path6]:
    os.makedirs(p, exist_ok=True)


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
    config.enable_stream(rs.stream.color,     W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth,     W, H, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8,  FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8,  FPS)
except RuntimeError as e:
    print(f"ストリームの設定に失敗しました: {e}")
    exit(0)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


# --- 3. メインループ (Enterで開始) ---
try:
    print("\nストリーミング準備完了。")
    print("プレビューウィンドウで [Enter] キーを押すと保存を開始します。")
    print("（[q] キーで保存せずに終了します）")

    is_running = True
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        images = np.vstack((np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap))))
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("保存を開始します... ([q]で停止)")
            break
        elif key == ord('q'):
            print("保存せずに終了します。")
            is_running = False
            break

    while is_running:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        images = np.vstack((np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap))))
        cv2.imshow('RealSense', images)

        cv2.imwrite(os.path.join(path1, f"{i}_color.jpg"), color_image)
        cv2.imwrite(os.path.join(path2, f"{i}_depth_colormap.jpg"), depth_colormap)
        cv2.imwrite(os.path.join(path3, f"{i}_ir_left.jpg"), ir_image1)
        cv2.imwrite(os.path.join(path4, f"{i}_ir_right.jpg"), ir_image2)
        cv2.imwrite(os.path.join(path5, f"{i}_ir_left_color.jpg"), ir_colormap1)
        cv2.imwrite(os.path.join(path6, f"{i}_ir_right_color.jpg"), ir_colormap2)

        i += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("保存を停止します。")
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
