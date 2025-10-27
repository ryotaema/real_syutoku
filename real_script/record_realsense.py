# mp4_collect_direct.py

import pyrealsense2 as rs # type: ignore
import numpy as np
import cv2 # type: ignore
import os
import time
import gc
from datetime import datetime

# === 保存ファイルの設定 ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/mp4/')
os.makedirs(save_dir, exist_ok=True)

# 1. 生データ(.bag)の保存パス
bag_path = os.path.join(save_dir, f'stream_{timestamp}.bag')
# 2. カラー映像(.mp4)の保存パス
mp4_path = os.path.join(save_dir, f'color_video_{timestamp}.mp4')


# === RealSense ストリーム設定 ===
W, H, FPS = 640, 480, 30 # 元のスクリプトの30FPS設定を維持
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

# 以前エラーの原因となったInfraredストリームは無効化します
# config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
# config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)

# 1. .bagファイルへの録画を有効化 (生データ用)
config.enable_record_to_file(bag_path)


# === MP4書き出し準備 (カラー映像用) ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))
if not video_writer.isOpened():
    print(f"❌ Failed to open VideoWriter for {mp4_path}")
    exit(1)


# === ストリーミング開始準備 ===
pipeline = rs.pipeline()
align_to = rs.stream.color
align = rs.align(align_to)

# === 録画開始待機 ===
print("--------------------------------------------------")
print(f"  生データ (BAG)  -> {bag_path}")
print(f"  カラー動画 (MP4) -> {mp4_path}")
print(f"  ターゲットFPS: {FPS}")
print("--------------------------------------------------")
try:
    input(">>> 準備完了。Enterキーを押すと録画を開始します...")
except KeyboardInterrupt:
    print("\nキャンセルされました。")
    video_writer.release()
    exit()

# === ストリーミングと録画の開始 ===
profile = pipeline.start(config)
print("--------------------------------------------------")
print(f"🟢 録画中... 'q' を押すと終了します")
print("--------------------------------------------------")

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

        # --- ★MP4ファイルへの書き込み★ ---
        video_writer.write(color_image)
        # ---------------------------------

        # --- 表示処理 ---
        # Depth画像
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # 表示
        images = np.hstack((color_image, depth_color_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # 'q'を押してウィンドウを閉じる
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🔴 録画停止... 'q' が押されました。")
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    
    # MP4ファイルを閉じる
    video_writer.release()
    print(f"✅ MP4ファイルの書き込み完了: {mp4_path}")
    
    cv2.destroyAllWindows()
    
    # .bagファイルのインデックス書き込みが完了するのを待機
    print("生データ(.bag)の最終処理（インデックス書き込み）を実行中...")
    del pipeline
    
    print(f"✅ .bagファイルの書き込み完了: {bag_path}")
    print("--------------------------------------------------")
    print("すべての処理が完了しました。")

# 変換スクリプトの案内を削除
# print("録画終了、.bagファイル保存完了,以下を実行してください")
# print(f"python3 convert_bag_to_mp4.py {bag_path}")