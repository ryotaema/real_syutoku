import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc

# カウンタ変数
i = 0
j = 1

# ベースとなる保存フォルダ
save_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/images/')
os.makedirs(save_dir, exist_ok=True)

# 新規 image_x フォルダを作成
while True:
    base_path = os.path.join(save_dir, f"image_{j}")
    try:
        os.makedirs(base_path)
        break
    except FileExistsError:
        j += 1

# サブディレクトリを一気に作成
paths = {
    'color':     os.path.join(base_path, 'color'),
    'depth':     os.path.join(base_path, 'depth'),
    'ir_left':   os.path.join(base_path, 'ir_left'),
    'ir_right':  os.path.join(base_path, 'ir_right'),
    'ir_l_col':  os.path.join(base_path, 'ir_left_color'),
    'ir_r_col':  os.path.join(base_path, 'ir_right_color'),
    'pc':        os.path.join(base_path, 'pointcloud'),
}
for p in paths.values():
    os.makedirs(p, exist_ok=True)

print("Save directory:", base_path)

# RealSense pipeline セットアップ
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)
pc    = rs.pointcloud()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        c_frame = aligned.get_color_frame()
        d_frame = aligned.get_depth_frame()
        ir1     = aligned.get_infrared_frame(1)
        ir2     = aligned.get_infrared_frame(2)
        if not c_frame or not d_frame:
            continue

        # 画像化
        color = np.asanyarray(c_frame.get_data())
        depth = np.asanyarray(d_frame.get_data())
        ir_l  = np.asanyarray(ir1.get_data())
        ir_r  = np.asanyarray(ir2.get_data())

        # カラーマップ（表示用）
        dm = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.4), cv2.COLORMAP_JET)
        ir_lc = cv2.applyColorMap(cv2.convertScaleAbs(ir_l), cv2.COLORMAP_JET)
        ir_rc = cv2.applyColorMap(cv2.convertScaleAbs(ir_r), cv2.COLORMAP_JET)

        # ウィンドウ表示
        top    = np.hstack((ir_lc, ir_rc))
        bottom = np.hstack((color, dm))
        disp   = np.vstack((top, bottom))
        cv2.imshow('RealSense', disp)

        # ファイル名ベース
        filename = f"{i:04d}"

        # 保存
        cv2.imwrite(os.path.join(paths['color'],    filename + "_color.jpg"),       color)
        cv2.imwrite(os.path.join(paths['depth'],    filename + "_depth_colormap.jpg"), dm)
        cv2.imwrite(os.path.join(paths['ir_left'],  filename + "_ir_left.jpg"),     ir_l)
        cv2.imwrite(os.path.join(paths['ir_right'], filename + "_ir_right.jpg"),    ir_r)
        cv2.imwrite(os.path.join(paths['ir_l_col'], filename + "_ir_left_color.jpg"),  ir_lc)
        cv2.imwrite(os.path.join(paths['ir_r_col'], filename + "_ir_right_color.jpg"), ir_rc)

        # 点群取得＆保存
        pc.map_to(c_frame)
        points = pc.calculate(d_frame)
        ply_path = os.path.join(paths['pc'], filename + "_pointcloud.ply")
        points.export_to_ply(ply_path, c_frame) 

        i += 1

        # キー操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
