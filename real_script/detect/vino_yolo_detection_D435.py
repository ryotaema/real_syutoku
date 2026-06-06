# ノートPC内蔵GPU(Intel Arc/Iris Xe)をOpenVINOで利用するスクリプト

from openvino.runtime import Core, Tensor
import pyrealsense2 as rs
import numpy as np
import cv2
import gc
import yaml
from pathlib import Path

with open(Path(__file__).parent.parent / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
_model_xml = str(_root / _cfg['model']['openvino_path'])
_conf_thresh = _cfg['model']['confidence_threshold']


def decode_yolo_output(output, image_shape, threshold=0.3):
    boxes = []
    if output.ndim == 3:
        output = output[0]
    output = output.T  # shape: (N, 5)

    for row in output:
        x1, y1, x2, y2, conf = row
        if conf < threshold:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return boxes

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x1, y1, x2, y2, conf) in boxes:
        label = f"{conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def get_realsense_color_frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())

# RealSenseの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

pipeline.start(config)
align = rs.align(rs.stream.color)

# OpenVINOモデルのロードとコンパイル
ie = Core()
model = ie.read_model(_model_xml)
compiled = ie.compile_model(model=model, device_name="GPU")

# 非同期推論の設定
num_requests = 2
infer_requests = [compiled.create_infer_request() for _ in range(num_requests)]
req_idx = 0

# 入力サイズ取得
_, _, iH, iW = compiled.input(0).shape

try:
    while True:
        color = get_realsense_color_frame(pipeline, align)
        if color is None:
            continue

        img = cv2.resize(color, (iW, iH))
        img_input = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        r = infer_requests[req_idx]
        r.set_input_tensor(0, Tensor(img_input))
        r.start_async()

        prev = infer_requests[(req_idx - 1) % num_requests]
        if prev.wait() == 0:
            out = prev.get_output_tensor(0).data.squeeze()
            boxes = decode_yolo_output(out, color.shape, threshold=_conf_thresh)
            draw_boxes(color, boxes)

        cv2.imshow('OpenVINO GPU', color)

        req_idx = (req_idx + 1) % num_requests
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
