# 連続フレームを自動保存するデータ収集スクリプト（D405/D435両対応）

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera, get_depth_alpha

_args = build_parser().parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

# --- カメラ検出 ---
try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir      = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# --- 保存フォルダ設定 ---
i = 0
j = 1
save_dir = os.path.expanduser(_cfg['output']['images_dir'])
date_str = datetime.now().strftime('%Y-%m-%d')
save_dir_dated = os.path.join(save_dir, date_str)
os.makedirs(save_dir_dated, exist_ok=True)

while True:
    base_path = os.path.join(save_dir_dated, f"image_{j}")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        break
    else:
        print("ファイルが存在しています: " + base_path)
        j += 1

path_color = os.path.join(base_path, "color/")
path_depth = os.path.join(base_path, "depth/")
paths_to_make = [path_color, path_depth]
if _has_ir:
    path_ir_left        = os.path.join(base_path, "ir_left/")
    path_ir_right       = os.path.join(base_path, "ir_right/")
    path_ir_left_color  = os.path.join(base_path, "ir_left_color/")
    path_ir_right_color = os.path.join(base_path, "ir_right_color/")
    paths_to_make += [path_ir_left, path_ir_right, path_ir_left_color, path_ir_right_color]

for path in paths_to_make:
    os.makedirs(path, exist_ok=True)

print("Save directory:", base_path)

# --- RealSense 設定 ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)
if _has_ir:
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image    = np.asanyarray(color_frame.get_data())
        depth_image    = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=_depth_alpha), cv2.COLORMAP_JET)

        if _has_ir:
            ir_frame1 = frames.get_infrared_frame(1)
            ir_frame2 = frames.get_infrared_frame(2)
            if not ir_frame1 or not ir_frame2:
                continue
            ir_image1    = np.asanyarray(ir_frame1.get_data())
            ir_image2    = np.asanyarray(ir_frame2.get_data())
            ir_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
            ir_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
            preview = np.vstack((
                np.hstack((color_image, depth_colormap)),
                np.hstack((ir_colormap1, ir_colormap2))
            ))
        else:
            preview = np.hstack((color_image, depth_colormap))

        cv2.imshow('RealSense View', preview)

        cv2.imwrite(os.path.join(path_color, f"{i:04d}_color.jpg"), color_image)
        cv2.imwrite(os.path.join(path_depth, f"{i:04d}_depth.jpg"), depth_colormap)
        if _has_ir:
            cv2.imwrite(os.path.join(path_ir_left,        f"{i:04d}_ir.jpg"),       ir_image1)
            cv2.imwrite(os.path.join(path_ir_right,       f"{i:04d}_ir.jpg"),       ir_image2)
            cv2.imwrite(os.path.join(path_ir_left_color,  f"{i:04d}_ir_color.jpg"), ir_colormap1)
            cv2.imwrite(os.path.join(path_ir_right_color, f"{i:04d}_ir_color.jpg"), ir_colormap2)

        i += 1
        print(f"\rsaved: {i} frames", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
