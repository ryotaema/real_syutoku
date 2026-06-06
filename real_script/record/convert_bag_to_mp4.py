# bagファイルをmp4に変換するスクリプト
# 使い方: python3 convert_bag_to_mp4.py <bagファイルパス>

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args

_args = build_parser(bag_input=True).parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

bag_path = os.path.expanduser(_args.bag_path)

if not os.path.exists(bag_path):
    print(f"指定された.bagファイルが見つかりません: {bag_path}")
    sys.exit(1)

filename = os.path.splitext(os.path.basename(bag_path))[0]
mp4_path = os.path.join(os.path.dirname(bag_path), f"{filename}.mp4")

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))

print(f"変換開始: {bag_path} → {filename}.mp4")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        video_writer.write(color_image)
except RuntimeError:
    print("最後まで再生されました。")
finally:
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"MP4出力完了: {mp4_path}")
