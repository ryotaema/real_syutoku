import cv2
from ultralytics import YOLO
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera
from camera import open_camera

_args = build_parser(include_model=True).parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
YOLO_MODEL_PATH = str(_root / _cfg['model']['yolo_path'])

# カメラSDKを初期化する前にQtの画面とイベント処理を準備する。
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.waitKey(1)

try:
    _cam = detect_camera(_args.serial)
except RuntimeError as e:
    print(f"エラー: {e}")
    cv2.destroyAllWindows()
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")

camera = open_camera(_cam, W, H, FPS, depth=False)
try:
    camera.start()
except RuntimeError as e:
    print(e)
    cv2.destroyAllWindows()
    exit(1)


if __name__ == '__main__':

    model = YOLO(YOLO_MODEL_PATH)
    # model.to("cuda")  # GPU使用時はコメントを外す

    try:
        while True:
            fr = camera.read()
            if fr is None:
                continue

            color_image = fr.color

            results = model(color_image, show=False, save=False)
            anotated_image = results[0].plot()

            cv2.imshow('RealSense', anotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        gc.collect()
