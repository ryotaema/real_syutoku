import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import gc
import yaml
from pathlib import Path

with open(Path(__file__).parent.parent / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
YOLO_MODEL_PATH = str(_root / _cfg['model']['yolo_path'])

ctx = rs.context()
serials = []
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()

if len(ctx.devices) > 0:
    for dev in ctx.devices:
        print('Found device:', dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
        serials.append(dev.get_info(rs.camera_info.serial_number))
else:
    print("No Intel Device connected")

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

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, show=False, save=False)
            anotated_image = results[0].plot()

            result_object = results[0]
            bounding_boxes = result_object.boxes.xyxy
            class_ids = result_object.boxes.cls
            class_names_dict = result_object.names

            for box, class_id in zip(bounding_boxes, class_ids):
                class_names_dict[int(class_id)]

            bbox = np.array([])
            for box in bounding_boxes:
                bbox = np.append(bbox, (float(box[0]), float(box[1])))
            bbox = bbox.reshape(len(bounding_boxes), 2)

            cv2.imshow('RealSense', anotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        del color_frame, color_image
        gc.collect()
