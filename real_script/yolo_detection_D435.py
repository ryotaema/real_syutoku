import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import gc

#学習済みモデルパス
YOLO_MODEL_PATH = '/home/ryota/annot_labelimg/real_syutoku/real_script/model/250626_weights/best.pt'


#デバイスの接続確認
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

# ストリームの設定
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# ストリームを設定
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# ストリーミング開始
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)



if __name__ == '__main__':

    # YOLOモデルのロード
    model = YOLO(YOLO_MODEL_PATH)
    
    # 推論に使用するプロセッサを指定(GPUある場合)
    #model.to("cuda")
    try:
        while True:
            # フレームセットを待機
            frames = pipeline.wait_for_frames()

            # フレームを取得
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            # Numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())

            # YOLOの処理
            results = model(color_image, show=False, save=False)

            anotated_image = results[0].plot()

            # results[0]からResultsオブジェクトを取り出す
            result_object = results[0]

            # バウンディングボックスの座標を取得
            bounding_boxes = result_object.boxes.xyxy

            # クラスIDを取得
            class_ids = result_object.boxes.cls

            # クラス名の辞書を取得
            class_names_dict = result_object.names

            # バウンディングボックスとクラス名を組み合わせて表示
            for box, class_id in zip(bounding_boxes, class_ids):
                class_name = class_names_dict[int(class_id)]
                # print(f"Box coordinates: {box}, Object: {class_name}")
            
            bbox = np.array([])
            for box in bounding_boxes:
                bbox = np.append(bbox, (float(box[0]), float(box[1])))
                
            bbox = bbox.reshape(len(bounding_boxes), 2)
            #print(bbox)
            # 推論結果表示
            cv2.imshow('RealSense', anotated_image)

            # 'q'を押してウィンドウを閉じる
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # ストリーミング停止
        pipeline.stop()
        cv2.destroyAllWindows()
        del color_frame, color_image
        gc.collect()
