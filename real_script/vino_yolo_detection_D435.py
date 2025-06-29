from openvino.runtime import Core, Tensor
import pyrealsense2 as rs
import numpy as np
import cv2
import gc

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

# Realsenseの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

# OpenVINOモデルのロードとコンパイル
ie = Core()
model = ie.read_model("model/250626_weights/openvino_model/best.xml")
compiled = ie.compile_model(model=model, device_name="GPU")

# 非同期推論の設定
num_requests = 2
infer_requests = [compiled.create_infer_request() for _ in range(num_requests)]
req_idx = 0

# 入力サイズ取得
_, _, H, W = compiled.input(0).shape

try:
    while True:
        # Realsenseから画像取得
        color = get_realsense_color_frame(pipeline, align)
        if color is None:
            continue

        # 画像前処理
        img = cv2.resize(color, (W, H))
        img_input = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # 推論開始
        r = infer_requests[req_idx]
        r.set_input_tensor(0, Tensor(img_input))  # ← 修正ここ！
        r.start_async()

        # 前回の推論結果を取得して描画
        prev = infer_requests[(req_idx - 1) % num_requests]
        
        if prev.wait() == 0:
            out = prev.get_output_tensor(0).data.squeeze()
            boxes = decode_yolo_output(out, color.shape, threshold=0.3)
            draw_boxes(color, boxes)

        cv2.imshow('OpenVINO GPU', color)


        req_idx = (req_idx + 1) % num_requests
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    gc.collect()
S