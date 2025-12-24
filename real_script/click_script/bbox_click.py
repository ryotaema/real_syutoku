import cv2
import os
import glob
import tkinter as tk
from tkinter import filedialog
import sys

# === GUIでフォルダ（imagesフォルダ）を選択 ===
def select_directory():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title="画像が含まれるフォルダ(images)を選択してください")
    root.destroy()
    if not dir_path:
        print("フォルダが選択されませんでした。終了します。")
        sys.exit()
    
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(image_paths), dir_path

def print_usage():
    print("\n" + "="*50)
    print("【操作方法】")
    print("  マウスドラッグ  : BBoxの追加")
    print("  マウスクリック  : 既存BBoxの選択（青枠になります）")
    print("  十字キー (↑↓←→) : 選択中のBBoxを1pxずつ微調整")
    print("-" * 50)
    print("  [s] : 手動保存 (Save)")
    print("  [z] : 1つ戻す (Undo)")
    print("  [c] : 削除モード ON/OFF (クリックした枠を消去)")
    print("  [d] : 次の画像へ (Next) ※自動保存されます")
    print("  [a] : 前の画像へ (Back)")
    print("  [q] : プログラムを終了 (Quit)")
    print("="*50 + "\n")

image_paths, img_dir = select_directory()
base_dir = os.path.dirname(img_dir)
label_class = 0

# === グローバル変数 ===
drawing = False
deleting = False
ix, iy = -1, -1
mx, my = -1, -1 
boxes = []
selected_idx = -1 
current_index = 0

def inside_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_bbox(event, x, y, flags, param):
    global ix, iy, mx, my, drawing, boxes, deleting, selected_idx
    mx, my = x, y
    if deleting:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(boxes):
                if inside_bbox(x, y, box):
                    del boxes[i]
                    selected_idx = -1
                    break
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_any = False
            for i, box in enumerate(boxes):
                if inside_bbox(x, y, box):
                    selected_idx = i
                    clicked_any = True
                    break
            if not clicked_any:
                drawing = True
                ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                boxes.append((x1, y1, x2, y2))
                selected_idx = len(boxes) - 1

# --- 保存・読み込み関連 ---
def get_label_path(image_path):
    filename = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(filename)[0]
    label_dir = os.path.join(base_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    return os.path.join(label_dir, f"{name_wo_ext}.txt")

def save_yolo_format(image_path, bboxes):
    img = cv2.imread(image_path)
    if img is None: return
    h, w, _ = img.shape
    txt_path = get_label_path(image_path)
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2 in bboxes:
            x_c, y_c = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            f.write(f"{label_class} {x_c:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n") # y_center は変数ミス修正 y_c へ
    print(f"✅ 保存完了: {os.path.basename(txt_path)}")

# 上記の save_yolo_format 内に一部タイポがあったため修正版
def save_yolo_format(image_path, bboxes):
    img = cv2.imread(image_path)
    if img is None: return
    h, w, _ = img.shape
    txt_path = get_label_path(image_path)
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2 in bboxes:
            x_c = (x1 + x2) / 2 / w
            y_c = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{label_class} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"✅ 保存完了: {os.path.basename(txt_path)}")

def load_yolo_format(image_path):
    txt_path = get_label_path(image_path)
    if not os.path.exists(txt_path): return []
    img = cv2.imread(image_path)
    if img is None: return []
    h, w, _ = img.shape
    loaded_boxes = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                _, xc, yc, bw, bh = map(float, parts)
                x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
                x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
                loaded_boxes.append((x1, y1, x2, y2))
    except: return []
    return loaded_boxes

def load_points(image_path):
    basename = os.path.basename(image_path)
    token = basename.replace("image_", "").replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    point_path = os.path.join(base_dir, "points", f"points_{token}.txt")
    points = []
    if os.path.exists(point_path):
        with open(point_path, "r") as f:
            for line in f:
                try:
                    x, y = map(int, line.strip().split(','))
                    points.append((x, y))
                except: continue
    return points

# === メインループ ===
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_bbox)

print_usage()

while current_index < len(image_paths):
    image_path = image_paths[current_index]
    original_image = cv2.imread(image_path)
    if original_image is None:
        current_index += 1
        continue
    
    boxes = load_yolo_format(image_path)
    points = load_points(image_path)
    selected_idx = -1 
    
    image_with_points = original_image.copy()
    for (px, py) in points:
        cv2.circle(image_with_points, (px, py), 5, (0, 255, 0), -1)

    print(f"--- 現在の画像 [{current_index + 1}/{len(image_paths)}]: {os.path.basename(image_path)}")

    while True:
        display = image_with_points.copy()
        h, w, _ = display.shape

        cv2.line(display, (0, my), (w, my), (200, 200, 200), 1)
        cv2.line(display, (mx, 0), (mx, h), (200, 200, 200), 1)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = (255, 0, 0) if i == selected_idx else (0, 255, 255)
            thickness = 3 if i == selected_idx else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        
        if drawing:
            cv2.rectangle(display, (ix, iy), (mx, my), (0, 0, 255), 2)

        title = f"[{current_index+1}/{len(image_paths)}] {os.path.basename(image_path)} {'[DELETE MODE]' if deleting else ''}"
        cv2.setWindowTitle('Image', title)
        cv2.imshow('Image', display)

        key = cv2.waitKey(20)

        # 十字キー微調整
        if selected_idx != -1:
            x1, y1, x2, y2 = boxes[selected_idx]
            if key == 81: # Left
                boxes[selected_idx] = (max(0, x1-1), y1, max(1, x2-1), y2)
            elif key == 82: # Up
                boxes[selected_idx] = (x1, max(0, y1-1), x2, max(1, y2-1))
            elif key == 83: # Right
                boxes[selected_idx] = (x1+1, y1, x2+1, y2)
            elif key == 84: # Down
                boxes[selected_idx] = (x1, y1+1, x2, y2+1)

        key = key & 0xFF
        if key == ord('s'):
            save_yolo_format(image_path, boxes)
        elif key == ord('z'):
            if boxes: 
                boxes.pop()
                selected_idx = -1
                print("Undo: 最後のBBoxを削除しました")
        elif key == ord('c'): # 削除モード切替
            deleting = not deleting
            print(f"削除モード: {'ON' if deleting else 'OFF'}")
        elif key == ord('d'): # 次の画像へ
            save_yolo_format(image_path, boxes)
            current_index += 1
            break
        elif key == ord('a'): # 前の画像へ
            if current_index > 0:
                current_index -= 1
                break
        elif key == ord('q'):
            print("プログラムを終了します。")
            sys.exit()

cv2.destroyAllWindows()