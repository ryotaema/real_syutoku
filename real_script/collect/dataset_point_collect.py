import pyrealsense2 as rs
import numpy as np
import cv2
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (load_config, build_parser, apply_args, detect_camera, get_depth_alpha,
                   make_depth_colormap, Session, cam_code)

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

_mods = ['color', 'depth_colormap', 'pointcloud']
if _has_ir:
    _mods += ['ir_left', 'ir_right', 'ir_left_color', 'ir_right_color']

session = Session(_cfg['output']['images_dir'], cam_code(_cam['model']),
                  tag=_args.tag, subdirs=_mods)
print("Save directory:", session.dir)

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
        dm    = make_depth_colormap(depth, _depth_alpha)

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

        i += 1
        cv2.imwrite(session.path(i, 'color'),          color)
        cv2.imwrite(session.path(i, 'depth_colormap'), dm)
        if _has_ir:
            cv2.imwrite(session.path(i, 'ir_left'),        ir_l)
            cv2.imwrite(session.path(i, 'ir_right'),       ir_r)
            cv2.imwrite(session.path(i, 'ir_left_color'),  ir_lc)
            cv2.imwrite(session.path(i, 'ir_right_color'), ir_rc)

        pc.map_to(c_frame)
        points = pc.calculate(d_frame)
        points.export_to_ply(session.path(i, 'pointcloud', ext='ply'), c_frame)

        print(f"\rsaved: {i} frames", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    session.write_metadata(
        camera={'name': _cam['name'], 'model': _cam['model'], 'serial': _cam['serial'],
                'resolution': [W, H], 'fps': FPS},
        modalities=_mods,
        shot_count=i,
    )
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
