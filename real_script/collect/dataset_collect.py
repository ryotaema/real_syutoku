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

# --- 1. カメラ検出 ---
try:
    _cam = detect_camera(_args.serial)
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir     = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# --- 2. 保存ディレクトリの設定 ---
i = 0

_mods = ['color', 'depth_colormap']
if _has_ir:
    _mods += ['ir_left', 'ir_right', 'ir_left_color', 'ir_right_color']

session = Session(_cfg['output']['images_dir'], cam_code(_cam['model']),
                  tag=_args.tag, subdirs=_mods)
print(f"画像を {session.dir} に保存します")


# --- 3. カメラの初期化 ---
camera = open_camera(_cam, W, H, FPS, ir=_has_ir)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    exit(1)


def _get_frames(fr):
    if fr is None:
        return None
    color_image    = fr.color
    depth_colormap = make_depth_colormap(fr.depth, _depth_alpha)
    ir_image1 = ir_image2 = ir_colormap1 = ir_colormap2 = None
    if _has_ir:
        ir_image1    = fr.ir_left
        ir_image2    = fr.ir_right
        ir_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
    return color_image, depth_colormap, ir_image1, ir_image2, ir_colormap1, ir_colormap2


def _make_preview(color_image, depth_colormap, ir_colormap1, ir_colormap2):
    if _has_ir:
        return np.vstack((
            np.hstack((ir_colormap1, ir_colormap2)),
            np.hstack((color_image,  depth_colormap))
        ))
    return np.hstack((color_image, depth_colormap))


# --- 4. メインループ (Enterで開始) ---
try:
    print("\nストリーミング準備完了。")
    print("プレビューウィンドウで [Enter] キーを押すと保存を開始します。")
    print("（[q] キーで保存せずに終了します）")

    is_running = True
    while True:
        result = _get_frames(camera.read())
        if result is None:
            continue
        color_image, depth_colormap, ir_image1, ir_image2, ir_colormap1, ir_colormap2 = result

        cv2.imshow('RealSense', _make_preview(color_image, depth_colormap, ir_colormap1, ir_colormap2))

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("保存を開始します... ([q]で停止)")
            break
        elif key == ord('q'):
            print("保存せずに終了します。")
            is_running = False
            break

    while is_running:
        result = _get_frames(camera.read())
        if result is None:
            continue
        color_image, depth_colormap, ir_image1, ir_image2, ir_colormap1, ir_colormap2 = result

        cv2.imshow('RealSense', _make_preview(color_image, depth_colormap, ir_colormap1, ir_colormap2))

        i += 1
        cv2.imwrite(session.path(i, 'color'),          color_image)
        cv2.imwrite(session.path(i, 'depth_colormap'), depth_colormap)
        if _has_ir:
            cv2.imwrite(session.path(i, 'ir_left'),        ir_image1)
            cv2.imwrite(session.path(i, 'ir_right'),       ir_image2)
            cv2.imwrite(session.path(i, 'ir_left_color'),  ir_colormap1)
            cv2.imwrite(session.path(i, 'ir_right_color'), ir_colormap2)

        print(f"\rsaved: {i} frames", end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n保存を停止します。")
            break

finally:
    print("ストリーミングを停止し、リソースを解放します。")
    session.write_metadata(
        camera=camera_metadata(_cam, W, H, FPS, camera),
        modalities=_mods,
        shot_count=i,
    )
    camera.stop()
    cv2.destroyAllWindows()
    gc.collect()
