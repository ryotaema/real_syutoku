# bagファイルに録画するスクリプト。変換は convert_bag_to_mp4.py を使用。

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args

_args = build_parser().parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_dir = os.path.expanduser(_cfg['output']['mp4_dir'])
os.makedirs(save_dir, exist_ok=True)
bag_path = os.path.join(save_dir, f'stream_{timestamp}.bag')

config = rs.config()
config.enable_stream(rs.stream.color,      W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth,      W, H, rs.format.z16,  FPS)
config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8,  FPS)
config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8,  FPS)
config.enable_record_to_file(bag_path)

pipeline = rs.pipeline()
profile = pipeline.start(config)
print("録画中... 'q' を押すと終了します")

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        images = np.hstack((color_image, depth_color_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("録画停止... 'q' が押されました。")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("ファイルの最終処理（インデックス書き込み）を実行中...")
    del pipeline

print("録画終了、.bagファイル保存完了。以下を実行してください:")
print(f"python3 convert_bag_to_mp4.py {bag_path}")
