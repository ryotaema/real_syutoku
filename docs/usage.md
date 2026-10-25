# 使い方詳細

[README](../README.md) の「使い方」の補足です．ファイル名の付け方・保存される形式は [data_format.md](data_format.md) を参照してください．

## ディレクトリ構成

```
real_syutoku/
├── real_script/
│   ├── config.yaml          # 解像度・FPS・出力先・モデルパスの設定
│   ├── config.local.yaml.example  # 環境ごとの上書き設定の例（config.local.yaml にコピーして使う）
│   ├── utils.py             # 共通ユーティリティ（設定読み込み・カメラ検出）
│   ├── camera.py            # カメラ取得の共通層（RealSense / Orbbec の差を吸収）
│   ├── gui.py               # 統合GUI（各スクリプトをボタンで起動）
│   ├── collect/             # 静止画データ収集
│   ├── record/              # 動画録画
│   ├── detect/              # リアルタイム推論
│   ├── process/             # 後処理（点群マージ・タイムラプス集計）
│   └── click_script/        # アノテーションツール
├── docs/                    # 詳しい説明（この資料など）
├── tools/
│   ├── rename_legacy.py     # 旧命名データを新命名規則へ変換（CLI）
│   └── rename_legacy_gui.py # 同上（GUI）
└── data/
    ├── images/              # 収集画像（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
    ├── mp4/                 # 録画ファイル
    ├── pointcloud/          # 点群データ（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
    └── timelapse_data/      # 定点タイムラプス（YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/）
```

## GUI

```bash
python3 real_script/gui.py
```

1. 上の「カメラ」に接続中のカメラが表示されます（2台以上つないでいるときはここで選びます）．
2. やりたいことのタブ（画像・点群・動画・タイムラプス・推論・アノテーション）を開き，ボタンを押します．
3. カメラ映像のウィンドウが開いたら，ボタン横の「操作」に書かれたキーで保存・終了します．
4. 途中でやめたいときは「停止」を押します（データは保存されたまま安全に終了します）．
5. 「保存先を開く」で，最後に使った機能の保存フォルダが開きます．

GUI は下記のスクリプトをボタンで起動しているだけなので，コマンドラインからの使い方も今までどおりです．
細かいオプション（解像度・FPS など）を変えたいときはコマンドラインか `config.yaml` を使ってください．
（OpenVINO 推論と .bag のみの録画 `mp4_collect.py` は GUI には入れていません．）

## RGB・深度・IR・PLYの一括取得

「画像」タブの **1枚ずつ撮る** は従来の画像保存で，PLYは計算・保存しません．
**全データを1枚ずつ撮る** は，`s` を押すたびにRGB・深度・IR・PLYなどを一括保存します（`q` で終了）．
自動で1枚取得するボタンはありません．ROSは不要です．

**全データを1枚ずつ撮る**（CLIでは `--all-data`）で保存される内容は [data_format.md](data_format.md#全データ取得dataset_collect_photo---all-data) を参照してください．

コマンドラインでは，リポジトリのルートから次のように実行できます．

```bash
# 通常の画像保存（PLYなし）。sで保存，qで終了
python3 real_script/collect/dataset_collect_photo.py

# PLYを含む全データをsで1組ずつ保存，qで終了
python3 real_script/collect/dataset_collect_photo.py --all-data

# 画面を使わず1ショット保存．待ち時間は任意に変更できる
python3 real_script/collect/dataset_collect_photo.py --all-data --once --no-preview --warmup 2
```

必要なフレームが揃わない場合は既定で10秒後にエラーにします（`--timeout` で変更可）．
選択した保存モードのファイルが揃ったショットだけを撮影枚数に数えます．
既存の同名ファイルは上書きせず，同じ秒に別セッションを開始する場合も保存先を分けます．

すべてのスクリプトは `real_script/` を起点に実行してください．

## CLI オプション

`config.yaml` を編集せずに，実行時だけ設定を上書きできます．

| オプション | 対象 | 例 |
|---|---|---|
| `--fps N` | collect / record / detect | `--fps 15` |
| `--width N` | collect / record / detect | `--width 1280` |
| `--height N` | collect / record / detect | `--height 720` |
| `--tag NAME` | collect / click_script | `--tag greenhouse`（セッションディレクトリ名にのみ付与） |
| `--serial SERIAL` | collect / record / detect / click_dataset | `--serial G336_SERIAL_NUMBER`（複数台接続時に使うカメラを指定．省略時は選択ダイアログ） |
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

## データ収集

```bash
# Enter で収集開始，q で停止
python3 collect/dataset_collect.py

# 点群（.ply）付き収集
python3 collect/dataset_point_collect.py

# ICP 用点群データ収集（intrinsics.json + 生深度付き）
python3 collect/pointcloud_capture.py                  # auto: 50フレーム自動取得
python3 collect/pointcloud_capture.py --frames 100     # フレーム数指定
python3 collect/pointcloud_capture.py --mode manual    # manual: [s]で1枚ずつ取得

# 1枚ずつ保存（s で保存，q で終了）
python3 collect/dataset_collect_photo.py
```

### 収集画像の保存先

セッションディレクトリ名は「カメラ・日付・開始時刻」で決まります（[データ命名規則](data_format.md#データ命名規則)）．

```
images_dir/YYMMDD/{cam}_{YYMMDD}_{HHMMSS}/
├── color/          {prefix}_{NNNNN}_c.jpg     # カラー画像（全機種共通）
├── depth_colormap/ {prefix}_{NNNNN}_dc.jpg    # 深度カラーマップ（全機種共通）
├── ir_left/        {prefix}_{NNNNN}_i1.jpg    # IR 左（D405 以外）
├── ir_right/       {prefix}_{NNNNN}_i2.jpg    # IR 右（D405 以外）
├── ir_left_color/  {prefix}_{NNNNN}_i1c.jpg   # IR 左カラーマップ（D405 以外）
├── ir_right_color/ {prefix}_{NNNNN}_i2c.jpg   # IR 右カラーマップ（D405 以外）
└── metadata.json                              # カメラ設定・撮影枚数
```

例: `images/260624/d435_260624_101741/color/d435_260624_101741_00001_c.jpg`

`dataset_point_collect.py` はこれに加えて `pointcloud/{prefix}_{NNNNN}_pc.ply` を保存します．

## 定点タイムラプス撮影

植物などの定点観察・時系列データ収集向けスクリプトです．5分間隔・12時間などの長時間無人撮影を想定しています．全機種対応で，D405 接続時は IR を除いた color と depth のみを保存します．

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
├── ir_left/        {prefix}_{NNNNN}_i1.jpg    # IR 左（D405 以外）
├── ir_right/       {prefix}_{NNNNN}_i2.jpg    # IR 右（D405 以外）
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

## 動画録画

全機種対応です．D405 接続時は IR ストリームを除いて録画します．

`.bag` は SDK ごとの形式で保存されます（RealSense の `.bag` と Orbbec の `.bag` は互換性がなく，
RealSense Viewer で Orbbec の `.bag` は開けません）．`convert_bag_to_mp4.py` はどちらの形式かを
自動で判別し，解像度・FPS も録画時の値を `.bag` から読み取ります．

```
mp4_dir/
├── {cam}_{YYMMDD}_{HHMMSS}_stream.bag   # 生データ（color / depth / IR）
├── {cam}_{YYMMDD}_{HHMMSS}_c.mp4        # カラー動画（record_realsense）
├── {cam}_{YYMMDD}_{HHMMSS}_det.mp4      # YOLO 描画済み動画（record_with_yolo）
└── {cam}_{YYMMDD}_{HHMMSS}_stream.mp4   # convert_bag_to_mp4 の出力
```

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

## 推論

```bash
# PyTorch（.pt モデル）
python3 detect/yolo_detection_D435.py

# OpenVINO（内蔵GPU使用）
source openvino_env/bin/activate
python3 detect/vino_yolo_detection_D435.py
```

## 点群合わせ込み・マージ

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

## アノテーション

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

## テスト

保存処理と模擬カメラによるテストは，カメラを接続せずに実行できます．

```bash
python3 -m unittest discover -s tests -p test_snapshot.py
```
