import pyrealsense2 as rs
import numpy as np
import cv2
import os
import gc

# --- 1. 保存ディレクトリの設定 ---
i = 0 # 保存する画像の連番
j = 1 # image_j フォルダの連番

# ベースとなる保存先
save_dir_base = os.path.expanduser('~/annot_labelimg/real_syutoku/data/images/')
os.makedirs(save_dir_base, exist_ok=True)

# 重複しない "image_j" フォルダを検索
file_ok = True
base_path = ""
while file_ok:
    base_path = os.path.join(save_dir_base, f"image_{j}")
    try:
        os.makedirs(base_path)
        file_ok = False # フォルダ作成に成功したらループを抜ける
    except FileExistsError:
        print(f"フォルダが存在しています: {base_path}")
        file_ok = True
    j += 1

# 各画像タイプの保存パスを設定
path1 = os.path.join(base_path, "color/")
path2 = os.path.join(base_path, "depth/")
path3 = os.path.join(base_path, "ir_left/")
path4 = os.path.join(base_path, "ir_right/")
path5 = os.path.join(base_path, "ir_left_color/")
path6 = os.path.join(base_path, "ir_right_color/")

print(f"画像を {base_path} に保存します")

# サブフォルダを作成
os.makedirs(path1, exist_ok=True)
os.makedirs(path2, exist_ok=True)
os.makedirs(path3, exist_ok=True)
os.makedirs(path4, exist_ok=True)
os.makedirs(path5, exist_ok=True)
os.makedirs(path6, exist_ok=True)


# --- 2. RealSenseの初期化 ---
pipeline = rs.pipeline()
config = rs.config()
ctx = rs.context()

# デバイス接続確認
devices = ctx.query_devices()
if len(devices) == 0:
    print("No Intel Device connected")
    exit(0)

for dev in devices:
    print('Found device:', dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
    # dev.hardware_reset() # 安定動作のため、通常はコメントアウトを推奨

# ストリームを設定
try:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15) # IR Left
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15) # IR Right
except RuntimeError as e:
    print(f"ストリームの設定に失敗しました: {e}")
    exit(0)

# ストリーミング開始
pipeline.start(config)

# アライメント（位置合わせ）の設定
align_to = rs.stream.color
align = rs.align(align_to)


# --- 3. メインループ (Enterで開始) ---
try:
    
    print("\nストリーミング準備完了。")
    print("プレビューウィンドウで [Enter] キーを押すと保存を開始します。")
    print("（[q] キーで保存せずに終了します）")

    # --- 3-1. Enterキーが押されるまで待機するループ (プレビューのみ) ---
    is_running = True
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        if not aligned_frames:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue

        # Numpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        # カラーマップに変換 (表示用)
        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        # 画像を連結して表示
        images = np.vstack(( np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap)) ))
        cv2.imshow('RealSense', images)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Enterキー (キーコード 13) が押されたら、このループを抜けて保存ループに進む
        if key == 13: 
            print("保存を開始します... ([q]で停止)")
            break
        # 'q'キーが押されたら、フラグを立ててループを抜ける
        elif key == ord('q'):
            print("保存せずに終了します。")
            is_running = False # 保存ループに入らないようにフラグを立てる
            break

    # --- 3-2. メインの保存ループ (is_running が True の場合のみ実行) ---
    while is_running:
        # フレームセットを待機
        frames = pipeline.wait_for_frames()

        # フレームを取得
        aligned_frames = align.process(frames)
        if not aligned_frames: continue # フレームが飛んだらスキップ
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            continue # どれか一つでもフレームが欠けていたらスキップ

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
        
        # --- 画像保存 ---
        cv2.imwrite(os.path.join(path1, f"{i}_color.jpg"), color_image)
        cv2.imwrite(os.path.join(path2, f"{i}_depth_colormap.jpg"), depth_colormap)
        cv2.imwrite(os.path.join(path3, f"{i}_ir_left.jpg"), ir_image1)
        cv2.imwrite(os.path.join(path4, f"{i}_ir_right.jpg"), ir_image2)
        cv2.imwrite(os.path.join(path5, f"{i}_ir_left_color.jpg"), ir_colormap1)
        cv2.imwrite(os.path.join(path6, f"{i}_ir_right_color.jpg"), ir_colormap2)

        i = i+1

        # 'q'を押してウィンドウを閉じる
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("保存を停止します。")
            break # この (is_running) ループを抜ける

finally:
    # --- 4. クリーンアップ ---
    print("ストリーミングを停止し、リソースを解放します。")
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # メモリを明示的に解放 (NameErrorを避けるためtry-except)
    try:
        del color_frame, color_image
        del depth_frame, depth_image
        del ir_frame1, ir_image1
        del ir_frame2, ir_image2
        del aligned_frames, frames
    except NameError:
        pass # 変数が定義される前に終了した場合
    
    gc.collect()  # ガベージコレクション（必要なら）