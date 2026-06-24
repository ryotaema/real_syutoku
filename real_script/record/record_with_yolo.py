import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, build_parser, apply_args, detect_camera

_args = build_parser(include_model=True).parse_args()
_cfg  = apply_args(load_config(), _args)

W   = _cfg['camera']['width']
H   = _cfg['camera']['height']
FPS = _cfg['camera']['fps']

_root = Path(__file__).parent.parent
MODEL_PATH = str(_root / _cfg['model']['yolo_path'])

print(f"Loading YOLO model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"指定されたモデルファイルが見つかりません: {MODEL_PATH}")
    print("config.yaml の model.yolo_path を確認してください。")
    exit(1)
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
save_dir = os.path.expanduser(_cfg['output']['mp4_dir'])
os.makedirs(save_dir, exist_ok=True)

bag_path = os.path.join(save_dir, f'stream_{timestamp}.bag')
mp4_path = os.path.join(save_dir, f'detected_{timestamp}.mp4')

try:
    _cam = detect_camera()
except RuntimeError as e:
    print(f"エラー: {e}")
    exit(1)
print(f"使用カメラ: {_cam['name']}  (シリアル: {_cam['serial']})")
_has_ir = (_cam['model'] != 'D405')

config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  FPS)
if _has_ir:
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
config.enable_record_to_file(bag_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))
if not video_writer.isOpened():
    print(f"Failed to open VideoWriter for {mp4_path}")
    exit(1)

pipeline = rs.pipeline()
align_to = rs.stream.color
align = rs.align(align_to)

print("--------------------------------------------------")
print(f"  モデル: {MODEL_PATH}")
print(f"  生データ (BAG)  -> {bag_path}")
print(f"  検出動画 (MP4)  -> {mp4_path}")
print("--------------------------------------------------")
try:
    input(">>> 準備完了。Enterキーを押すと録画を開始します...")
except KeyboardInterrupt:
    print("\nキャンセルされました。")
    video_writer.release()
    sys.exit(0)

profile = pipeline.start(config)
print("録画中... 'q' を押すと終了します")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, verbose=False)
        annotated_frame = results[0].plot()
        video_writer.write(annotated_frame)

        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        images = np.hstack((annotated_frame, depth_color_image))
        cv2.namedWindow('RealSense with YOLO', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense with YOLO', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("録画停止... 'q' が押されました。")
            break

finally:
    pipeline.stop()
    video_writer.release()
    print(f"MP4ファイルの書き込み完了: {mp4_path}")
    cv2.destroyAllWindows()
    print("生データ(.bag)の最終処理（インデックス書き込み）を実行中...")
    del pipeline
    print(f".bagファイルの書き込み完了: {bag_path}")
    print("すべての処理が完了しました。")
