import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc

#画像を保存するフォルダ名

i = 0
j = 1

#scriptディレクトリのひとつ上
save_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/images/')
os.makedirs(save_dir, exist_ok=True)
#os.chdir('..')
#print(os.getcwd())
file_ok = True

while file_ok:
    base_path = os.path.join(save_dir, f"image_{j}")
    try:
        #base_path = os.getcwd() + "/image_" + str(j)
        os.makedirs(base_path)
        file_ok = False
    except FileExistsError:
        print("ファイルが存在しています"+base_path)
        file_ok = True
    j = j+1

#パスの設定
path1 = base_path + "/color/"
path2 = base_path + "/depth/"
path3 = base_path + "/ir_left/"
path4 = base_path + "/ir_right/"
path5 = base_path + "/ir_left_color/"
path6 = base_path + "/ir_right_color/"

print("Save directory: " + base_path)

os.makedirs(path1)
os.makedirs(path2)
os.makedirs(path3)
os.makedirs(path4)
os.makedirs(path5)
os.makedirs(path6)


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

# print("press [s] to start")
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#             print("start")
#             break

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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)


# ストリーミング開始
pipeline.start(config)

align_to = rs.stream.color
#align_to = rs.stream.depth
align = rs.align(align_to)

try:
    while True:
        # フレームセットを待機
        frames = pipeline.wait_for_frames()

        # フレームを取得
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        # Numpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        # カラーマップに変換
        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)#alpha=0.08

        # カラーとデプス画像を並べて表示
        images = np.vstack(( np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap)) ))
        cv2.imshow('RealSense', images)
        cv2.imwrite(path1+str(i)+"color.jpg",color_image)
        cv2.imwrite(path2+str(i)+"depth_colormap.jpg",depth_colormap)
        cv2.imwrite(path3+str(i)+"ir_left.jpg",ir_image1)
        cv2.imwrite(path4+str(i)+"ir_right.jpg",ir_image2)
        cv2.imwrite(path5+str(i)+"ir_left_color.jpg",ir_colormap1)
        cv2.imwrite(path6+str(i)+"ir_right_color.jpg",ir_colormap2)


        i = i+1

        # 'q'を押してウィンドウを閉じる
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()
    del color_frame, color_image  # NumPy配列などを明示的に消す
    gc.collect()  # ガベージコレクション（必要なら）
