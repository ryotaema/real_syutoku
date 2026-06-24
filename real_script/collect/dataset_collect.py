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

# --- 1. カメラ検出 ---
try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir     = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# --- 2. 保存ディレクトリの設定 ---
i = 0

save_dir_base = os.path.expanduser(_cfg['output']['images_dir'])
_now     = datetime.now()
date_key = _now.strftime('%Y_%m%d')   # 例: 2026_0624
date_str = _now.strftime('%Y-%m-%d')  # 例: 2026-06-24
time_str = _now.strftime('%H%M%S')    # 例: 101741
save_dir_dated = os.path.join(save_dir_base, date_key)
os.makedirs(save_dir_dated, exist_ok=True)

existing_sessions = [d for d in os.scandir(save_dir_dated) if d.is_dir() and d.name.startswith('image')]
N = len(existing_sessions) + 1
base_path = os.path.join(save_dir_dated, f"image{N}_{date_str}_{time_str}_{_cam['model']}")

path_color = os.path.join(base_path, "color")
path_depth = os.path.join(base_path, "depth")
paths_to_make = [path_color, path_depth]
if _has_ir:
    path_ir_left        = os.path.join(base_path, "ir_left")
    path_ir_right       = os.path.join(base_path, "ir_right")
    path_ir_left_color  = os.path.join(base_path, "ir_left_color")
    path_ir_right_color = os.path.join(base_path, "ir_right_color")
    paths_to_make += [path_ir_left, path_ir_right, path_ir_left_color, path_ir_right_color]

print(f"画像を {base_path} に保存します")
for p in paths_to_make:
    os.makedirs(p, exist_ok=True)


# --- 3. RealSenseの初期化 ---
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


def _get_frames(aligned_frames):
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color_image    = np.asanyarray(color_frame.get_data())
    depth_image    = np.asanyarray(depth_frame.get_data())
    depth_colormap = make_depth_colormap(depth_image, _depth_alpha)
    ir_image1 = ir_image2 = ir_colormap1 = ir_colormap2 = None
    if _has_ir:
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)
        if not ir_frame1 or not ir_frame2:
            return None
        ir_image1    = np.asanyarray(ir_frame1.get_data())
        ir_image2    = np.asanyarray(ir_frame2.get_data())
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
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        result = _get_frames(aligned_frames)
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
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        result = _get_frames(aligned_frames)
        if result is None:
            continue
        color_image, depth_colormap, ir_image1, ir_image2, ir_colormap1, ir_colormap2 = result

        cv2.imshow('RealSense', _make_preview(color_image, depth_colormap, ir_colormap1, ir_colormap2))

        cv2.imwrite(os.path.join(path_color, f"{i}_color.jpg"),          color_image)
        cv2.imwrite(os.path.join(path_depth, f"{i}_depth_colormap.jpg"), depth_colormap)
        if _has_ir:
            cv2.imwrite(os.path.join(path_ir_left,        f"{i}_ir_left.jpg"),        ir_image1)
            cv2.imwrite(os.path.join(path_ir_right,       f"{i}_ir_right.jpg"),       ir_image2)
            cv2.imwrite(os.path.join(path_ir_left_color,  f"{i}_ir_left_color.jpg"),  ir_colormap1)
            cv2.imwrite(os.path.join(path_ir_right_color, f"{i}_ir_right_color.jpg"), ir_colormap2)

        i += 1
        print(f"\rsaved: {i} frames", end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n保存を停止します。")
            break

finally:
    print("ストリーミングを停止し、リソースを解放します。")
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
