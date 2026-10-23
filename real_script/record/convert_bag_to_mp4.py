# bagファイルをmp4に変換するスクリプト（RealSense / Orbbec の .bag は自動判別）
# 使い方: python3 convert_bag_to_mp4.py <bagファイルパス>
# 解像度・FPS は録画時の値を .bag から読み取る

import cv2
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import build_parser
from camera import bag_backend, open_playback

_args = build_parser(bag_input=True).parse_args()

bag_path = os.path.expanduser(_args.bag_path)

if not os.path.exists(bag_path):
    print(f"指定された.bagファイルが見つかりません: {bag_path}")
    sys.exit(1)

filename = os.path.splitext(os.path.basename(bag_path))[0]
mp4_path = os.path.join(os.path.dirname(bag_path), f"{filename}.mp4")

try:
    player = open_playback(bag_path)
    player.start()
except RuntimeError as e:
    print(f"エラー: {e}")
    sys.exit(1)
if player.size is None:
    print("color フレームが含まれていません。")
    player.stop()
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, player.fps, player.size)

print(f"変換開始: {bag_path} → {filename}.mp4  "
      f"({bag_backend(bag_path)}, {player.size[0]}x{player.size[1]} @ {player.fps}fps)")

n = 0
try:
    while not player.eof:
        fr = player.read()
        if fr is None:
            continue
        video_writer.write(fr.color)
        n += 1
    print(f"最後まで再生されました。（{n} フレーム）")
finally:
    player.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"MP4出力完了: {mp4_path}")
