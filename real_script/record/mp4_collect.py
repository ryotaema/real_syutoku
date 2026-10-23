# bagファイルに録画するスクリプト。変換は convert_bag_to_mp4.py を使用。

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
_has_ir      = (_cam['model'] != 'D405')
_depth_alpha = get_depth_alpha(_cfg, _cam['model'])

# 動画はショット連番を持たないため {cam}_{YYMMDD}_{HHMMSS}_{種別}.{ext} で命名する
save_dir = os.path.expanduser(_cfg['output']['mp4_dir'])
os.makedirs(save_dir, exist_ok=True)
prefix   = make_prefix(cam_code(_cam['model']))
bag_path = os.path.join(save_dir, f'{prefix}_stream.bag')

camera = open_camera(_cam, W, H, FPS, ir=_has_ir, record_path=bag_path)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    exit(1)
print("録画中... 'q' を押すと終了します")

try:
    while True:
        fr = camera.read()
        if fr is None:
            continue

        images = np.hstack((fr.color, make_depth_colormap(fr.depth, _depth_alpha)))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("録画停止... 'q' が押されました。")
    print("ファイルの最終処理（インデックス書き込み）を実行中...")
    camera.stop()
    cv2.destroyAllWindows()

print("録画終了、.bagファイル保存完了。以下を実行してください:")
print(f"python3 convert_bag_to_mp4.py {bag_path}")
