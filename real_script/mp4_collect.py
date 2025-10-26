#mp4の録画を出力するために作成したスクリプト
#出力はbagファイルのため，別途作成している変換用のスクリプトを利用してください．

import pyrealsense2 as rs # type: ignore
import numpy as np
import cv2 # type: ignore
import os
import time
import gc
from datetime import datetime


# 保存ファイルの設定
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/mp4/')
os.makedirs(save_dir, exist_ok=True)
bag_path = os.path.join(save_dir, f'stream_{timestamp}.bag')

# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_record_to_file(bag_path)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
print("録画中... 'q' を押すと終了します")

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        # Depth画像
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # 表示
        images = np.hstack((color_image, depth_color_image))
        #images = np.hstack((color_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # 'q'を押してウィンドウを閉じる
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # ストリーミング停止
    #pipeline.stop()
    #cv2.destroyAllWindows()
    print("録画停止... 'q' が押されました。")
    pipeline.stop()
    cv2.destroyAllWindows()
    
    print("ファイルの最終処理（インデックス書き込み）を実行中...")
    del pipeline
    
    #del color_frame, color_image
    #gc.collect()
    #print("録画終了。ファイルの書き込みが完了するまで待機中...")
    #time.sleep(5)

print("録画終了、.bagファイル保存完了,以下を実行してください")
print(f"python3 convert_bag_to_mp4.py {bag_path}")
