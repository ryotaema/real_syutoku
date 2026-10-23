import numpy as np
import cv2
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (load_config, build_parser, apply_args, detect_camera, make_prefix, cam_code,
                   get_depth_alpha, make_depth_colormap)
from camera import open_camera

_args = build_parser().parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

try:
    _cam = detect_camera(_args.serial)
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# 動画はショット連番を持たないため {cam}_{YYMMDD}_{HHMMSS}_{種別}.{ext} で命名する
save_dir = os.path.expanduser(_cfg['output']['mp4_dir'])
os.makedirs(save_dir, exist_ok=True)
prefix   = make_prefix(cam_code(_cam['model']))

bag_path = os.path.join(save_dir, f'{prefix}_stream.bag')
mp4_path = os.path.join(save_dir, f'{prefix}_c.mp4')

camera = open_camera(_cam, W, H, FPS, record_path=bag_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))
if not video_writer.isOpened():
    print(f"Failed to open VideoWriter for {mp4_path}")
    exit(1)

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
    sys.exit(0)

try:
    camera.start()
except RuntimeError as e:
    print(e)
    video_writer.release()
    sys.exit(1)
print("録画中... 'q' を押すと終了します")

try:
    while True:
        fr = camera.read()
        if fr is None:
            continue

        color_image = fr.color
        video_writer.write(color_image)

        images = np.hstack((color_image, make_depth_colormap(fr.depth, _depth_alpha)))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("録画停止... 'q' が押されました。")
            break

finally:
    video_writer.release()
    print(f"MP4ファイルの書き込み完了: {mp4_path}")
    cv2.destroyAllWindows()
    print("生データ(.bag)の最終処理（インデックス書き込み）を実行中...")
    camera.stop()
    print(f".bagファイルの書き込み完了: {bag_path}")
    print("すべての処理が完了しました。")
