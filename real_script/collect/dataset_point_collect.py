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

i = 0
j = 1

save_dir = os.path.expanduser(_cfg['output']['images_dir'])
date_str = datetime.now().strftime('%Y-%m-%d')
save_dir_dated = os.path.join(save_dir, date_str)
os.makedirs(save_dir_dated, exist_ok=True)

while True:
    base_path = os.path.join(save_dir_dated, f"image_{j}")
    try:
        os.makedirs(base_path)
        break
    except FileExistsError:
        j += 1

paths = {
    'color': os.path.join(base_path, 'color'),
    'depth': os.path.join(base_path, 'depth'),
    'pc':    os.path.join(base_path, 'pointcloud'),
}
if _has_ir:
    paths['ir_left']  = os.path.join(base_path, 'ir_left')
    paths['ir_right'] = os.path.join(base_path, 'ir_right')
    paths['ir_l_col'] = os.path.join(base_path, 'ir_left_color')
    paths['ir_r_col'] = os.path.join(base_path, 'ir_right_color')

for p in paths.values():
    os.makedirs(p, exist_ok=True)

print("Save directory:", base_path)

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)
if _has_ir:
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)

pipeline.start(config)
align = rs.align(rs.stream.color)
pc    = rs.pointcloud()

try:
    while True:
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)

        c_frame = aligned.get_color_frame()
        d_frame = aligned.get_depth_frame()
        if not c_frame or not d_frame:
            continue

        color = np.asanyarray(c_frame.get_data())
        depth = np.asanyarray(d_frame.get_data())
        dm    = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=_depth_alpha), cv2.COLORMAP_JET)

        if _has_ir:
            ir1 = aligned.get_infrared_frame(1)
            ir2 = aligned.get_infrared_frame(2)
            if not ir1 or not ir2:
                continue
            ir_l  = np.asanyarray(ir1.get_data())
            ir_r  = np.asanyarray(ir2.get_data())
            ir_lc = cv2.applyColorMap(cv2.convertScaleAbs(ir_l), cv2.COLORMAP_JET)
            ir_rc = cv2.applyColorMap(cv2.convertScaleAbs(ir_r), cv2.COLORMAP_JET)
            preview = np.vstack((np.hstack((ir_lc, ir_rc)), np.hstack((color, dm))))
        else:
            preview = np.hstack((color, dm))

        cv2.imshow('RealSense', preview)

        filename = f"{i:04d}"
        cv2.imwrite(os.path.join(paths['color'], filename + "_color.jpg"),          color)
        cv2.imwrite(os.path.join(paths['depth'], filename + "_depth_colormap.jpg"), dm)
        if _has_ir:
            cv2.imwrite(os.path.join(paths['ir_left'],  filename + "_ir_left.jpg"),        ir_l)
            cv2.imwrite(os.path.join(paths['ir_right'], filename + "_ir_right.jpg"),       ir_r)
            cv2.imwrite(os.path.join(paths['ir_l_col'], filename + "_ir_left_color.jpg"),  ir_lc)
            cv2.imwrite(os.path.join(paths['ir_r_col'], filename + "_ir_right_color.jpg"), ir_rc)

        pc.map_to(c_frame)
        points = pc.calculate(d_frame)
        points.export_to_ply(os.path.join(paths['pc'], filename + "_pointcloud.ply"), c_frame)

        i += 1
        print(f"\rsaved: {i} frames", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
