import numpy as np
import cv2
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (load_config, build_parser, apply_args, detect_camera, get_depth_alpha,
                   make_depth_colormap, Session, cam_code, camera_metadata)
from camera import open_camera

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

camera = open_camera(_cam, W, H, FPS, ir=_has_ir)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    exit(1)

try:
    while True:
        fr = camera.read()
        if fr is None:
            continue

        color = fr.color
        dm    = make_depth_colormap(fr.depth, _depth_alpha)

        if _has_ir:
            ir_l  = fr.ir_left
            ir_r  = fr.ir_right
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

        camera.save_ply(fr, session.path(i, 'pointcloud', ext='ply'))

        print(f"\rsaved: {i} frames", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    session.write_metadata(
        camera=camera_metadata(_cam, W, H, FPS, camera),
        modalities=_mods,
        shot_count=i,
    )
    camera.stop()
    cv2.destroyAllWindows()
    gc.collect()
