# real_syutoku

Intel RealSense カメラを用いた画像取得・アノテーション・推論モデル検証用スクリプト群．

## ディレクトリ構成

```
real_syutoku/
├── real_script/
│   ├── config.yaml          # 解像度・FPS・出力先・モデルパスの設定
│   ├── utils.py             # 共通ユーティリティ（設定読み込み・カメラ検出）
│   ├── collect/             # 静止画データ収集
│   ├── record/              # 動画録画
│   ├── detect/              # リアルタイム推論
│   ├── process/             # 後処理（点群マージ・タイムラプス集計）
│   └── click_script/        # アノテーションツール
└── data/
    ├── images/              # 収集画像（YYYY_MMDD/imageN_YYYY-MM-DD_HHMMSS_CAMERA/）
    ├── mp4/                 # 録画ファイル
    ├── pointcloud/          # 点群データ（YYYY_MMDD/pcN_YYYY-MM-DD_HHMMSS_CAMERA/）
    └── timelapse_data/      # 定点タイムラプス（YYYY_MMDD/timelapsN_YYYY-MM-DD_HHMMSS_CAMERA/）
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

## カメラ自動認識

すべてのスクリプトは起動時に接続されているカメラを自動検出し，**D435・D405 どちらでも動作**します．

```
使用カメラ: Intel RealSense D435  (シリアル: 12345678)
```

| | D435 | D405 |
|---|---|---|
| color | ✓ | ✓ |
| depth | ✓ | ✓ |
| ir_left / ir_right | ✓ | なし |
| pointcloud | ✓ | ✓ |
| 深度カラーマップ | alpha=0.4 固定（~637mm 飽和） | フレーム内相対正規化 |

D405 接続時は IR ストリームを無効化し，color と depth のみを収集します．深度カラーマップは D435 が絶対距離を色で表す固定 alpha 方式，D405 がフレーム内の最近〜最遠を 0〜255 に正規化する相対方式に自動切り替えされます（D405 は距離レンジが狭く固定 alpha では飽和しやすいため）．

## 設定ファイル

解像度・FPS・出力先・モデルパスは `real_script/config.yaml` で一元管理しています．

```yaml
camera:
  width: 640
  height: 480
  fps: 30  # D435で4ストリーム同時使用時は15を推奨
  depth_alpha:
    D435: 0.4   # ~637mm で飽和
    D405: ~     # null → フレーム内相対正規化
    default: 0.4

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

`config.yaml` を編集せずに，実行時だけ設定を上書きできます．

| オプション | 対象 | 例 |
|---|---|---|
| `--fps N` | collect / record / detect | `--fps 15` |
| `--width N` | collect / record / detect | `--width 1280` |
| `--height N` | collect / record / detect | `--height 720` |
| `--model PATH` | record_with_yolo / detect | `--model /path/to/model.pt` |
| `--conf F` | vino_yolo_detection / timelapse_detect（--detect 時） | `--conf 0.5` |
| `--interval N` | timelapse_detect | `--interval 300` |
| `--duration N` | timelapse_detect | `--duration 12` |
| `--detect` | timelapse_detect | `--detect` |
| `--relative-depth` | timelapse_detect | `--relative-depth` |

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
# Enter で収集開始，q で停止（D435/D405 両対応）
python3 collect/dataset_collect.py

# 点群（.ply）付き収集（D435/D405 両対応）
python3 collect/dataset_point_collect.py

# ICP 用点群データ収集（intrinsics.json + 生深度付き）
python3 collect/pointcloud_capture.py                  # auto: 50フレーム自動取得
python3 collect/pointcloud_capture.py --frames 100     # フレーム数指定
python3 collect/pointcloud_capture.py --mode manual    # manual: [s]で1枚ずつ取得

# 1枚ずつ保存（s で保存，q で終了）（D435/D405 両対応）
python3 collect/dataset_collect_photo.py
```

#### 収集画像の保存先

セッション名には「連番・日付・開始時刻・カメラモデル」が含まれます．同じ日に複数回実行すると連番が増えます．

```
images_dir/YYYY_MMDD/imageN_YYYY-MM-DD_HHMMSS_CAMERA/
├── color/          # カラー画像（D435/D405 共通）
├── depth/          # 深度画像（D435/D405 共通）
├── ir_left/        # IR 左（D435 のみ）
├── ir_right/       # IR 右（D435 のみ）
├── ir_left_color/  # IR 左カラーマップ（D435 のみ）
└── ir_right_color/ # IR 右カラーマップ（D435 のみ）
```

例: `images/2026_0624/image1_2026-06-24_101741_D435/`

### 定点タイムラプス撮影

植物などの定点観察・時系列データ収集向けスクリプトです．5分間隔・12時間などの長時間無人撮影を想定しています．D435/D405 両対応で，D405 接続時は IR を除いた color と depth のみを保存します．

```bash
# 基本（5分ごとに保存）
python3 collect/timelapse_detect.py

# 間隔・時間を変更
python3 collect/timelapse_detect.py --interval 600 --duration 6

# YOLO検出を追加（起動時にGUIでモデルを選択）
python3 collect/timelapse_detect.py --detect

# 深度カラーマップを相対値で表示（D435 で目視確認向け。D405 は常に相対表示）
python3 collect/timelapse_detect.py --relative-depth
```

各セッションは `data/timelapse_data/YYYY_MMDD/timelapsN_YYYY-MM-DD_HHMMSS_CAMERA/` に保存されます．

```
timelapse_data/YYYY_MMDD/timelapsN_YYYY-MM-DD_HHMMSS_CAMERA/
├── color/          NNNN_HHMMSS_color.jpg          # カラー画像
├── depth/          NNNN_HHMMSS_depth.png           # 16bit 生深度
│                   NNNN_HHMMSS_depth_colormap.jpg  # 深度可視化
├── ir_left/        NNNN_HHMMSS_ir_left.jpg         # IR 左（D435 のみ）
├── ir_right/       NNNN_HHMMSS_ir_right.jpg        # IR 右（D435 のみ）
├── annotated/      NNNN_HHMMSS_annotated.jpg       # BBOX付き（--detect 時のみ）
└── detection_log.csv                               # 検出結果ログ（--detect 時のみ）
```

例: `timelapse_data/2026_0624/timelapse1_2026-06-24_080000_D435/`

撮影後に認識率の時系列グラフを生成できます．

```bash
# 最新セッションを自動選択
python3 process/timelapse_analysis.py

# セッション指定
python3 process/timelapse_analysis.py data/timelapse_data/2026_0624/timelapse1_2026-06-24_080000_D435
```

出力: `<session_dir>/analysis.png`（検出数・平均信頼度の時系列グラフ）

### 動画録画

D435/D405 両対応です．D405 接続時は IR ストリームを除いて録画します．

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

### 点群合わせ込み・マージ

点群セッションは `data/pointcloud/YYYY_MMDD/pcN_YYYY-MM-DD_HHMMSS_CAMERA/` に保存されます．

```bash
# 最新セッションを自動選択
python3 process/point_merge.py

# カメラ固定・物体静止（全フレーム→frame0に位置合わせ）
python3 process/point_merge.py data/pointcloud/2026_0606/pc1_2026-06-06_120000_D435

# カメラ固定・物体回転（frame-to-frame逐次位置合わせ）
python3 process/point_merge.py data/pointcloud/2026_0606/pc1_2026-06-06_120000_D435 --sequential

# パラメータ上書き
python3 process/point_merge.py <session_dir> --voxel-size 0.003 --icp-threshold 0.01
```

パスは `real_script/` からの相対パス（`data/pointcloud/...`）と絶対パスの両方が使えます．

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

## ライセンス

このリポジトリは [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) のもとで公開されています。

本プロジェクトは [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）を使用しているため、AGPL-3.0 に従い同ライセンスを適用しています。
