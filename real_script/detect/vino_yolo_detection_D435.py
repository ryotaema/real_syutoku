# ノートPC内蔵GPU(Intel Arc/Iris Xe)をOpenVINOで利用するスクリプト

from openvino.runtime import Core, Tensor
import numpy as np
import cv2
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera
from camera import open_camera

_args = build_parser(include_model=True, include_conf=True).parse_args()
_cfg  = apply_args(load_config(), _args, model_key='openvino_path')

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
_model_xml = str(_root / _cfg['model']['openvino_path'])
_conf_thresh = _cfg['model']['confidence_threshold']


def decode_yolo_output(output, threshold=0.3):
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

def get_color_frame(camera):
    fr = camera.read()
    return None if fr is None else fr.color

# カメラの初期化
try:
    _cam = detect_camera(_args.serial)
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")

camera = open_camera(_cam, W, H, FPS, depth=False)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    exit(1)

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
        color = get_color_frame(camera)
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
            boxes = decode_yolo_output(out, threshold=_conf_thresh)
            draw_boxes(color, boxes)

        cv2.imshow('OpenVINO GPU', color)

        req_idx = (req_idx + 1) % num_requests
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
    gc.collect()
