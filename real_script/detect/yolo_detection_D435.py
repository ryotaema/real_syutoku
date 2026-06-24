import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera

_args = build_parser(include_model=True).parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
YOLO_MODEL_PATH = str(_root / _cfg['model']['yolo_path'])

try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


if __name__ == '__main__':

    model = YOLO(YOLO_MODEL_PATH)
    # model.to("cuda")  # GPU使用時はコメントを外す

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, show=False, save=False)
            anotated_image = results[0].plot()

            cv2.imshow('RealSense', anotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        del color_frame, color_image
        gc.collect()
