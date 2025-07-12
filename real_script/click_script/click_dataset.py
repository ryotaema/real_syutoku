#テストdataを収集するために画像と対象の座標

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc
from datetime import datetime

# 保存ディレクトリ
base_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/click_test_data')
os.makedirs(base_dir, exist_ok=True)

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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
print("マウスで検出対象をクリックしてください。's' で保存、'q' で終了")

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense', mouse_callback)

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
            # 保存処理
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            image_path = os.path.join(base_dir, f"image_{timestamp}.jpg")
            txt_path = os.path.join(base_dir, f"points_{timestamp}.txt")

            cv2.imwrite(image_path, color_image)
            with open(txt_path, 'w') as f:
                for pt in click_points:
                    f.write(f"{pt[0]},{pt[1]}\n")

            print(f"✅ 保存しました: {image_path}, {txt_path}")
            click_points.clear()  # 次の保存に備えてクリア

        elif key == ord('q'):
            break


finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
