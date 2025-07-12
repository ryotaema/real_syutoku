import cv2
import os
import glob

# === 設定 ===
base_dir = os.path.expanduser('~/annot_labelimg/real_syutoku/data/click_test_data/')
image_paths = sorted(glob.glob(os.path.join(base_dir, 'image_*.jpg')))
label_class = 0  # 必要に応じて変更

# === グローバル変数 ===
drawing = False
ix, iy = -1, -1
boxes = []
current_image = None
current_index = 0

# === マウスイベントコールバック ===
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_image = current_image.copy()
        for box in boxes:
            cv2.rectangle(temp_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow('Image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))
            print(f"追加: bbox=({x1}, {y1}) - ({x2}, {y2})")

# === YOLO形式で保存 ===
def save_yolo_format(image_path, bboxes):
    h, w, _ = cv2.imread(image_path).shape
    yolo_path = image_path.replace(".jpg", ".txt")
    with open(yolo_path, "w") as f:
        for x1, y1, x2, y2 in bboxes:
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h
            f.write(f"{label_class} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
    print(f"✅ YOLOアノテーションを保存: {yolo_path}")

# === YOLO形式を読み込む ===
def load_yolo_format(image_path):
    txt_path = image_path.replace(".jpg", ".txt")
    if not os.path.exists(txt_path):
        return []
    
    h, w, _ = cv2.imread(image_path).shape
    loaded_boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, bbox_w, bbox_h = map(float, parts)
            x_center *= w
            y_center *= h
            bbox_w *= w
            bbox_h *= h
            x1 = int(x_center - bbox_w / 2)
            y1 = int(y_center - bbox_h / 2)
            x2 = int(x_center + bbox_w / 2)
            y2 = int(y_center + bbox_h / 2)
            loaded_boxes.append((x1, y1, x2, y2))
    return loaded_boxes

# === メインループ ===
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_bbox)

for image_path in image_paths:
    boxes.clear()
    image = cv2.imread(image_path)
    current_image = image.copy()

    # 既存アノテーションがあれば読み込む
    boxes = load_yolo_format(image_path)

    # 対応するクリック点を描画
    timestamp = os.path.basename(image_path).replace("image_", "").replace(".jpg", "")
    point_path = os.path.join(base_dir, f"points_{timestamp}.txt")
    if os.path.exists(point_path):
        with open(point_path, "r") as f:
            for line in f:
                x, y = map(int, line.strip().split(','))
                cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)

    print(f"\n🖼️ 現在の画像: {image_path}")
    print("操作: マウスドラッグ→bbox追加, 's'→保存, 'n'→次画像, 'q'→終了")

    while True:
        display = current_image.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow('Image', display)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            save_yolo_format(image_path, boxes)
        elif key == ord('n'):
            break
        elif key == ord('q'):
            exit()

cv2.destroyAllWindows()
