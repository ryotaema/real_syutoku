#bagファイルをmp4に変換するスクリプト

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys

# === 引数から.bagファイルを指定 ===
if len(sys.argv) < 2:
    print("使い方: python3 convert_bag_to_mp4.py <bagファイルパス>")
    sys.exit(1)

bag_path = os.path.expanduser(sys.argv[1])

if not os.path.exists(bag_path):
    print(f"❌ 指定された.bagファイルが見つかりません: {bag_path}")
    sys.exit(1)

# === 出力ファイル名 ===
filename = os.path.splitext(os.path.basename(bag_path))[0]
mp4_path = os.path.join(os.path.dirname(bag_path), f"{filename}.mp4")

# === RealSense再生設定 ===
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# === MP4書き出し準備 ===
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, fps, (640, 480))

print(f"変換開始: {bag_path} → {filename}.mp4")
#print("再生ウィンドウを開いています。'q' を押すと中断できます。")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        video_writer.write(color_image)

        #cv2.imshow("Playback", color_image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    print("🛑 手動で再生を中断しました。")
        #    break
except RuntimeError:
    print("最後まで再生されました。")
finally:
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ MP4出力完了: {mp4_path}")
