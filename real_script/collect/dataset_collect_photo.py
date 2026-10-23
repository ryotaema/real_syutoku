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

print("\nストリーミング準備中...")

# --- 2. カメラの初期化 ---
try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir      = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

_mods = ['color', 'depth_colormap']
if _has_ir:
    _mods += ['ir_left', 'ir_right', 'ir_left_color', 'ir_right_color']

session = Session(_cfg['output']['images_dir'], cam_code(_cam['model']),
                  tag=_args.tag, subdirs=_mods)
print(f"画像を {session.dir} に保存します")

camera = open_camera(_cam, W, H, FPS, ir=_has_ir)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    exit(1)

shot_count = 0

# --- 3. メインループ ---
try:
    print("\n[s] キー : 画像を1枚保存")
    print("[q] キー : 終了\n")

    while True:
        fr = camera.read()
        if fr is None:
            continue

        color_image    = fr.color
        depth_colormap = make_depth_colormap(fr.depth, _depth_alpha)

        if _has_ir:
            ir_image1    = fr.ir_left
            ir_image2    = fr.ir_right
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
            shot_count += 1

            cv2.imwrite(session.path(shot_count, 'color'),          color_image)
            cv2.imwrite(session.path(shot_count, 'depth_colormap'), depth_colormap)

            if _has_ir:
                cv2.imwrite(session.path(shot_count, 'ir_left'),        ir_image1)
                cv2.imwrite(session.path(shot_count, 'ir_right'),       ir_image2)
                cv2.imwrite(session.path(shot_count, 'ir_left_color'),  ir_colormap1)
                cv2.imwrite(session.path(shot_count, 'ir_right_color'), ir_colormap2)

            print(f"[{shot_count}枚目保存] {session.name(shot_count, 'color')}")

        elif key == ord('q'):
            print(f"\n終了します。合計 {shot_count} 枚保存しました。")
            break

finally:
    print("ストリーミングを停止し、リソースを解放します。")
    session.write_metadata(
        camera=camera_metadata(_cam, W, H, FPS, camera),
        modalities=_mods,
        shot_count=shot_count,
    )
    camera.stop()
    cv2.destroyAllWindows()
    gc.collect()
