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

_mods = ['color', 'depth_colormap']
if _has_ir:
    _mods += ['ir_left', 'ir_right', 'ir_left_color', 'ir_right_color']

session = Session(_cfg['output']['images_dir'], cam_code(_cam['model']),
                  tag=_args.tag, subdirs=_mods)
print(f"画像を {session.dir} に保存します")

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

shot_count = 0

# --- 3. メインループ ---
try:
    print("\n[s] キー : 画像を1枚保存")
    print("[q] キー : 終了\n")

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
        camera={'name': _cam['name'], 'model': _cam['model'], 'serial': _cam['serial'],
                'resolution': [W, H], 'fps': FPS},
        modalities=_mods,
        shot_count=shot_count,
    )
    pipeline.stop()
    cv2.destroyAllWindows()

    try:
        del color_frame, color_image
        del depth_frame, depth_image
        del aligned_frames, frames
    except NameError:
        pass

    gc.collect()
