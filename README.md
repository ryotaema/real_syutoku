# real_syutoku

Intel RealSense カメラを用いた画像取得・アノテーション・推論モデル検証用スクリプト群．

## ディレクトリ構成

```
real_syutoku/
├── real_script/
│   ├── config.yaml          # 解像度・FPS・出力先・モデルパスの設定
│   ├── collect/             # 静止画データ収集
│   ├── record/              # 動画録画
│   ├── detect/              # リアルタイム推論
│   └── click_script/        # アノテーションツール
└── data/
    ├── images/              # 収集画像（YYYY-MM-DD/image_N/）
    └── mp4/                 # 録画ファイル
```

## セットアップ

```bash
pip install -r real_script/requirements.txt

# ICP合わせ込みスクリプト使用時のみ（強いPCで実行する場合も同様）
pip install -r real_script/requirements_processing.txt

# アノテーションツール使用時のみ
sudo apt install python3-tk
```

OpenVINOを使う場合は `openvino_env/` の仮想環境を使用してください．

## 設定ファイル

解像度・FPS・出力先・モデルパスは `real_script/config.yaml` で一元管理しています．スクリプトを実行する前に必要に応じて編集してください．

```yaml
camera:
  width: 640
  height: 480
  fps: 30  # D435で4ストリーム同時使用時は15を推奨

output:
  images_dir: ~/annot_labelimg/real_syutoku/data/images
  mp4_dir: ~/annot_labelimg/real_syutoku/data/mp4

model:
  yolo_path: model/260217_pepper_yolov11x_aug.pt
  openvino_path: model/250626_weights/openvino_model/best.xml
  confidence_threshold: 0.3
```

## 使い方

すべてのスクリプトは `real_script/` を起点に実行してください．

### CLI オプション

`config.yaml` を編集せずに、実行時だけ設定を上書きできます．

| オプション | 対象 | 例 |
|---|---|---|
| `--fps N` | collect / record / detect | `--fps 15` |
| `--width N` | collect / record / detect | `--width 1280` |
| `--height N` | collect / record / detect | `--height 720` |
| `--model PATH` | record_with_yolo / detect | `--model /path/to/model.pt` |
| `--conf F` | vino_yolo_detection のみ | `--conf 0.5` |

```bash
# D435 で 4 ストリームを 15 FPS で収集
python3 collect/dataset_collect.py --fps 15

# 別モデルで推論
python3 detect/yolo_detection_D435.py --model model/other_model.pt

# OpenVINO の信頼度を上げて推論
python3 detect/vino_yolo_detection_D435.py --conf 0.5
```

### データ収集

```bash
# D435：Enter で収集開始，q で停止
python3 collect/dataset_collect.py

# D405
python3 collect/d405_dataset_collect.py

# D435 + 点群（.ply）
python3 collect/dataset_point_collect.py

# ICP用点群データ収集（intrinsics.json + 生深度付き）
python3 collect/pointcloud_capture.py                  # auto: 50フレーム自動取得
python3 collect/pointcloud_capture.py --frames 100     # フレーム数指定
python3 collect/pointcloud_capture.py --mode manual    # manual: [s]で1枚ずつ取得

# 1枚ずつ保存（s で保存，q で終了）
python3 collect/dataset_collect_photo.py
```

収集画像は `images_dir/YYYY-MM-DD/image_N/{color, depth, ir_left, ...}/` に保存されます．

### 動画録画

```bash
# .bag 形式で録画（終了後に変換コマンドが表示されます）
python3 record/mp4_collect.py

# .bag + .mp4 を同時録画
python3 record/record_realsense.py

# .bag + YOLO検出済み .mp4 を同時録画
python3 record/record_with_yolo.py

# .bag → .mp4 変換
python3 record/convert_bag_to_mp4.py <bagファイルのパス>
```

### 推論

```bash
# PyTorch（.pt モデル）
python3 detect/yolo_detection_D435.py

# OpenVINO（内蔵GPU使用）
source openvino_env/bin/activate
python3 detect/vino_yolo_detection_D435.py
```

### ICP点群合わせ込み

```bash
# カメラ固定・物体静止（全フレーム→frame0に位置合わせ）
python3 process/icp_merge.py data/pointcloud/2026-06-06/session_120000

# カメラ固定・物体回転（frame-to-frame逐次位置合わせ）
python3 process/icp_merge.py data/pointcloud/2026-06-06/session_120000 --sequential

# パラメータ上書き
python3 process/icp_merge.py <session_dir> --voxel-size 0.003 --icp-threshold 0.01
```

出力: `<session_dir>/merged_pointcloud.ply` と変換行列 `icp_result.json`

### アノテーション

```bash
# テストデータ作成（クリックで座標記録）
python3 click_script/click_dataset.py

# バウンディングボックスアノテーション（YOLO形式 .txt 出力）
python3 click_script/bbox_click.py
```

**bbox_click.py の操作：**
| キー/操作 | 動作 |
|---|---|
| 左ドラッグ（空白） | BBox新規作成 |
| 左ドラッグ（角） | BBoxリサイズ |
| 左ドラッグ（内部） | BBox移動 |
| [d] / [a] | 次/前の画像（[d]は自動保存） |
| [s] | 保存 |
| [c] | 前画像のBBoxをコピー |
| [z] | 直前のBBoxを取り消し |
| [Delete] | 選択中のBBoxを削除 |
| 矢印キー | 選択中のBBoxを1px移動 |

## OpenVINO モデル変換

```bash
yolo export model=best.pt format=onnx
ovc best.onnx --output_model openvino_model/best.xml
```

## RealSense セットアップ

Intel RealSense SDK (librealsense) を使用しています．
公式リポジトリ: https://github.com/IntelRealSense/librealsense

```bash
pip install pyrealsense2
```
