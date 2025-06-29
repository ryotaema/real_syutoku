import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc

# 保存フォルダ設定
i = 0
j = 1
save_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/images/')
os.makedirs(save_dir, exist_ok=True)

while True:
    base_path = os.path.join(save_dir, f"image_{j}")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        break
    else:
        print("ファイルが存在しています: " + base_path)
        j += 1

# 各画像保存先ディレクトリ作成
path_color = os.path.join(base_path, "color/")
path_depth = os.path.join(base_path, "depth/")
path_ir_left = os.path.join(base_path, "ir_left/")
path_ir_right = os.path.join(base_path, "ir_right/")
path_ir_left_color = os.path.join(base_path, "ir_left_color/")
path_ir_right_color = os.path.join(base_path, "ir_right_color/")

for path in [path_color, path_depth, path_ir_left, path_ir_right, path_ir_left_color, path_ir_right_color]:
    os.makedirs(path, exist_ok=True)

print("Save directory:", base_path)

# RealSense 設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # カラー追加
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        ir_frame1 = frames.get_infrared_frame(1)
        ir_frame2 = frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue

        # NumPy配列変換
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        # カラーマップ変換
        ir_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        # 表示
        images = np.vstack((
            np.hstack((color_image, depth_colormap)),
            np.hstack((ir_colormap1, ir_colormap2))
        ))
        cv2.imshow('D405 View', images)

        # 保存
        cv2.imwrite(os.path.join(path_color,        f"{i:04d}_color.jpg"), color_image)
        cv2.imwrite(os.path.join(path_depth,        f"{i:04d}_depth.jpg"), depth_colormap)
        cv2.imwrite(os.path.join(path_ir_left,      f"{i:04d}_ir.jpg"),    ir_image1)
        cv2.imwrite(os.path.join(path_ir_right,     f"{i:04d}_ir.jpg"),    ir_image2)
        cv2.imwrite(os.path.join(path_ir_left_color,  f"{i:04d}_ir_color.jpg"), ir_colormap1)
        cv2.imwrite(os.path.join(path_ir_right_color, f"{i:04d}_ir_color.jpg"), ir_colormap2)

        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
