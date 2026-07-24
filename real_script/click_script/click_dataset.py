#テストdataを収集するために画像と対象の座標

import pyrealsense2 as rs
import numpy as np
import cv2
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera, Session, cam_code

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
    sys.exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")

# 保存ディレクトリ（画像とクリック座標をペアで置くためセッション直下にフラット配置）
_base_dir = Path(_cfg['output']['images_dir']).expanduser().parent / 'click_test_data'
session   = Session(_base_dir, cam_code(_cam['model']), tag=_args.tag)
print(f"保存先: {session.dir}")

# グローバル変数
click_points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global click_points, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"クリック: ({x}, {y})")
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)

# RealSense設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)#size

pipeline.start(config)
print("マウスで検出対象をクリックしてください。's' で保存、'q' で終了")

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense', mouse_callback)

shot_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        current_frame = color_image.copy()

        # 前にクリックされた点を再描画
        for pt in click_points:
            cv2.circle(current_frame, pt, 5, (0, 255, 0), -1)

        cv2.imshow('RealSense', current_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 保存処理（画像とクリック座標は同じ連番を共有する）
            shot_count += 1
            image_path = session.path(shot_count, 'color', sub=False)
            txt_path   = session.path(shot_count, 'points', ext='txt', sub=False)

            cv2.imwrite(image_path, color_image)
            with open(txt_path, 'w') as f:
                for pt in click_points:
                    f.write(f"{pt[0]},{pt[1]}\n")

            print(f"保存しました: {image_path}, {txt_path}")
            click_points.clear()  # 次の保存に備えてクリア

        elif key == ord('q'):
            break


finally:
    session.write_metadata(
        camera={'name': _cam['name'], 'model': _cam['model'], 'serial': _cam['serial'],
                'resolution': [W, H], 'fps': FPS},
        modalities=['color', 'points'],
        shot_count=shot_count,
    )
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
