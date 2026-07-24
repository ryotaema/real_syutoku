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
├── tools/
│   ├── rename_legacy.py     # 旧命名データを新命名規則へ変換（CLI）
│   └── rename_legacy_gui.py # 同上（GUI）
└── data/
    ├── images/              # 収集画像（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
    ├── mp4/                 # 録画ファイル
    ├── pointcloud/          # 点群データ（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
    └── timelapse_data/      # 定点タイムラプス（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
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

## データ命名規則

すべての収集データは，ファイル名だけで「どのカメラの・いつの・何枚目の・何の画像か」が
分かるように統一されています．学習用に1つのフォルダへ集約しても，アノテーションツールに
まとめて読み込ませても，どのセッションのデータか判別できなくなることがありません．

```
{cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.{ext}

d435_260707_101741_00042_c.jpg
 │        │      │      │     └─ モダリティコード
 │        │      │      └─────── セッション内のショット連番（撮影時刻ではない）
 │        │      └────────────── セッション開始時刻
 │        └───────────────────── 取得日
 └────────────────────────────── カメラコード
```

| 要素 | 説明 |
|------|------|
| `cam` | `d435` / `d405`（自動判別）．nyx660_syutoku 側は `nyx` を使うため，リポジトリを跨いでも衝突しません |
| `YYMMDD` | 取得日 |
| `HHMMSS` | セッション開始時刻．1プロセス＝1セッションで固定です |
| `NNNNN` | セッション内のショット連番．**同一ショットの color/depth/ir は同じ番号を共有します** |
| `mod` | モダリティコード（下表） |

**モダリティコード**

| コード | 内容 | コード | 内容 |
|--------|------|--------|------|
| `c` | color | `i1c` | IR 左カラーマップ |
| `d` | depth（16bit raw PNG） | `i2c` | IR 右カラーマップ |
| `dc` | depth colormap | `pc` | 点群（.ply） |
| `i1` | IR 左 | `pt` | クリック座標（.txt） |
| `i2` | IR 右 | `det` | YOLO 描画済み |

連番はショット単位で共有されるため，末尾のモダリティコードを差し替えるだけで対応する
ファイルを引けます（`..._00042_c.jpg` ↔ `..._00042_dc.jpg`）．YOLO のラベル `.txt` や
labelImg の `.xml` も stem が一致するので自動的に紐づきます．

撮影条件など，ファイル名に載せない情報は各セッションの `metadata.json` に記録されます．

### セッションディレクトリとタグ

```
<出力先>/<YYMMDD>/<prefix>[_<tag>]/<モダリティ>/<ファイル>
```

`--tag` を付けると，セッションディレクトリ名にだけ任意の名前が付きます
（ファイル名の桁は増えません）．

```bash
python3 collect/dataset_collect_photo.py --tag greenhouse
# → images/260707/d435_260707_101741_greenhouse/color/d435_260707_101741_00001_c.jpg
```

### 旧データの変換

2026年7月より前に取得した旧命名（`imageN_YYYY-MM-DD_HHMMSS_D435/` など）のデータは，
そのまま置いておけます．新命名に揃えたい場合は変換ツールを使ってください．

旧ファイル名のモダリティ接尾辞（`_color` / `_depth_colormap` など）を手がかりに，
同一ショットのファイルへ同じ連番を振り直します．`labels/*.txt` や `*.xml` も追従します．
既定はコピーなので原データは残ります．

#### GUI（推奨）

フォルダを選んで，変換後のファイル名を一覧で確認してから実行できます．
データが複数の場所に散らばっている場合も，変換元フォルダをいくつでも登録できます．

```bash
python3 tools/rename_legacy_gui.py
```

1. **変換元フォルダ** — 「フォルダを追加...」で登録（複数可）．
   データルート（`data/images`）を指定すれば配下のセッションをまとめて拾います．
2. **設定** — カメラコードは通常「自動判別」のままで構いません．フォルダ名から
   判別できないデータ（`click_test_data` など）はスキップ理由が表示されるので，
   そのときだけ `d435` / `d405` を指定します．タグ・出力先・コピー/移動もここで指定．
3. **プレビューを作成** — セッションごとに「変換前 → 変換後」が並びます．
   行を開くと個々のファイル名を確認できます．
4. **変換を実行** — プレビューを作るまで実行ボタンは押せません．

`tkinter` が必要です（`sudo apt install python3-tk`）．

#### CLI

```bash
# 何が起きるか確認（dry-run。既定ではファイルを書き換えません）
python3 tools/rename_legacy.py data/images

# 実行（コピーで出力するため原データは残ります）
python3 tools/rename_legacy.py data/images --apply

# ディレクトリ名からカメラ・日時が分からないデータ
python3 tools/rename_legacy.py data/click_test_data/250911_testdata_click --cam d435 --apply
```

`--move` を付けると移動になります（原データが残らないので注意）．

## 使い方

すべてのスクリプトは `real_script/` を起点に実行してください．

### CLI オプション

`config.yaml` を編集せずに，実行時だけ設定を上書きできます．

| オプション | 対象 | 例 |
|---|---|---|
| `--fps N` | collect / record / detect | `--fps 15` |
| `--width N` | collect / record / detect | `--width 1280` |
| `--height N` | collect / record / detect | `--height 720` |
| `--tag NAME` | collect / click_script | `--tag greenhouse`（セッションディレクトリ名にのみ付与） |
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

セッションディレクトリ名は「カメラ・日付・開始時刻」で決まります（[データ命名規則](#データ命名規則)）．

```
images_dir/YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/
├── color/          {prefix}_{NNNNN}_c.jpg     # カラー画像（D435/D405 共通）
├── depth_colormap/ {prefix}_{NNNNN}_dc.jpg    # 深度カラーマップ（D435/D405 共通）
├── ir_left/        {prefix}_{NNNNN}_i1.jpg    # IR 左（D435 のみ）
├── ir_right/       {prefix}_{NNNNN}_i2.jpg    # IR 右（D435 のみ）
├── ir_left_color/  {prefix}_{NNNNN}_i1c.jpg   # IR 左カラーマップ（D435 のみ）
├── ir_right_color/ {prefix}_{NNNNN}_i2c.jpg   # IR 右カラーマップ（D435 のみ）
└── metadata.json                              # カメラ設定・撮影枚数
```

例: `images/260624/d435_260624_101741/color/d435_260624_101741_00001_c.jpg`

`dataset_point_collect.py` はこれに加えて `pointcloud/{prefix}_{NNNNN}_pc.ply` を保存します．

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

各セッションは `data/timelapse_data/YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/` に保存されます．

```
timelapse_data/YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/
├── color/          {prefix}_{NNNNN}_c.jpg     # カラー画像
├── depth/          {prefix}_{NNNNN}_d.png     # 16bit 生深度
├── depth_colormap/ {prefix}_{NNNNN}_dc.jpg    # 深度可視化
├── ir_left/        {prefix}_{NNNNN}_i1.jpg    # IR 左（D435 のみ）
├── ir_right/       {prefix}_{NNNNN}_i2.jpg    # IR 右（D435 のみ）
├── annotated/      {prefix}_{NNNNN}_det.jpg   # BBOX付き（--detect 時のみ）
├── detection_log.csv                          # 検出結果ログ（--detect 時のみ）
└── metadata.json                              # 撮影間隔・継続時間・カメラ設定
```

例: `timelapse_data/260624/d435_260624_080000/color/d435_260624_080000_00001_c.jpg`

連番は5桁なので，長時間のタイムラプス（10秒間隔×24時間＝8640枚）でも桁が溢れません．

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

点群セッションは `data/pointcloud/YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/` に保存されます
（color / depth / .ply をセッション直下にフラット配置）．

```bash
# 最新セッションを自動選択
python3 process/point_merge.py

# カメラ固定・物体静止（全フレーム→frame0に位置合わせ）
python3 process/point_merge.py data/pointcloud/260606/d435_260606_120000

# カメラ固定・物体回転（frame-to-frame逐次位置合わせ）
python3 process/point_merge.py data/pointcloud/260606/d435_260606_120000 --sequential

# パラメータ上書き
python3 process/point_merge.py <session_dir> --voxel-size 0.003 --icp-threshold 0.01
```

パスは `real_script/` からの相対パス（`data/pointcloud/...`）と絶対パスの両方が使えます．

出力: `<session_dir>/{prefix}_merged_pc.ply` と変換行列 `merge_result.json`

旧命名（`0000_pointcloud.ply`）のセッションもそのまま読めます．

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
