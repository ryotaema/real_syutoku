# セットアップ詳細

[README](../README.md) の「セットアップ」の補足です．

## Python パッケージ

PythonとメーカーSDKを使ってカメラへ直接接続します．ROSの導入やROS2ドライバのビルドは不要です．
リポジトリのルートで次を実行します．

```bash
pip install -r real_script/requirements.txt

# ICP合わせ込みスクリプト使用時のみ（強いPCで実行する場合も同様）
pip install -r real_script/requirements_processing.txt

# アノテーションツール使用時のみ
sudo apt install python3-tk
```

## RealSense セットアップ

Intel RealSense SDK (librealsense) を使用しています．
公式リポジトリ: https://github.com/IntelRealSense/librealsense

```bash
pip install pyrealsense2
```

## Orbbec セットアップ（Gemini 336）

Orbbec SDK v2 の Python バインディング（pyorbbecsdk）を使用しています．
公式リポジトリ: https://github.com/orbbec/pyorbbecsdk

```bash
pip install pyorbbecsdk2
```

初めて使う PC では udev ルールの設定が必要です（pyorbbecsdk リポジトリの `scripts/env_setup/install_udev_rules.sh`）．
pyrealsense2 / pyorbbecsdk は使う方だけ入っていれば動きます．

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
    G336: 0.4   # Gemini 336 も depth は mm 単位
    default: 0.4

output:                  # config.yaml からの相対パス（~ や絶対パスも可）
  images_dir: ../data/images
  pointcloud_dir: ../data/pointcloud
  timelapse_dir: ../data/timelapse_data
  mp4_dir: ../data/mp4

model:
  yolo_path: model/best.pt
  openvino_path: model/250626_weights/openvino_model/best.xml
  confidence_threshold: 0.3
```

出力先は既定でリポジトリ内の `data/` です（`config.yaml` からの相対パスなので，リポジトリをどこに clone しても
そのリポジトリの `data/` に保存されます）．クリック座標データは `images_dir` と同じ階層の `click_test_data/` に保存されます．

### 環境ごとの設定（config.local.yaml）

手元のモデル名や保存先など，自分の環境だけで変えたい値は `config.yaml` を直接編集せず，
`real_script/config.local.yaml` に書いてください．このファイルは git 管理外で，存在すれば `config.yaml` にキー単位で上書きされます．

```bash
cp real_script/config.local.yaml.example real_script/config.local.yaml
```

```yaml
# real_script/config.local.yaml（変えたいキーだけを書く）
model:
  yolo_path: model/my_model.pt
```

優先順位は「CLI オプション > `config.local.yaml` > `config.yaml`」です．相対パスの基準は `config.yaml` と同じ `real_script/` です．

## Docker環境内で利用する場合（任意）

Dockerを使う場合は，このリポジトリをコンテナにマウントし，Python依存もコンテナ内に導入します．
GUI用の画面共有，日本語フォント，カメラへアクセスするためのUSB設定も必要です．
用意済みのHumbleベースのDocker環境には，以下の設定と基本依存を含めています．
このプロジェクトの実行にROS2ドライバのビルドは必要ありません．

Docker構成ディレクトリから，起動時と同じCompose設定でコンテナに入ります．
以下はCPU構成の例です．NVIDIA構成では
`docker compose -f compose.yaml -f compose.nvidia.yaml exec humble bash` を使います．

```bash
docker compose exec humble bash
```

共有ワークスペース構成では，コンテナ内のワークスペースルートから次を実行します．

```bash
cd workspaces/real_syutoku
python3 real_script/gui.py
```

独自にHumbleベースのイメージを用意する場合は，以下をコンテナ内で実行します．
同じ環境に含まれる `cv_bridge` との互換性のため，NumPyとOpenCVに制約を付けています．

```bash
# コンテナ内の real_syutoku/ で実行
python3 -m pip install 'pip>=24,<26' 'packaging>=24,<26' 'setuptools<80' wheel
# Ubuntu 標準の blinker は削除せず，新しい版を追加する
python3 -m pip install --ignore-installed --no-deps 'blinker>=1.9,<2'
python3 -m pip install 'numpy<2' 'opencv-python<4.12' -r real_script/requirements.txt
```

Docker構成ディレクトリの `Dockerfile` には，この基本依存と日本語フォント
（`fonts-noto-cjk`），Tk，UTF-8 ロケール，USB 確認用の `usbutils` を設定しています．
日本語が四角で表示される場合は，コンテナ内で
`fc-list :lang=ja family` に日本語フォントが出るか確認し，導入後に GUI を起動し直してください．

カメラが見つからない場合は，まずコンテナ内で `lsusb` を実行し，RealSense または Orbbec が
見えるか確認します．このアプリの対象は RealSense / Orbbec の深度カメラで，PC 内蔵の
Web カメラは対象外です．GUI の「再検出」を押すと，SDK の未導入・読み込みエラーも下のログに
表示されます．現在の Compose は `/dev:/dev` と `privileged: true` を設定済みです．
ホストでの Orbbec の USB 権限設定は，上の「Orbbec セットアップ」を参照してください．

このDocker構成の推論にはCPU版またはNVIDIA版のPyTorchとUltralyticsを使用します．
OpenVINOは任意機能で，このDockerイメージには含めていません．

「保存先を開く」には，コンテナ内にもファイルマネージャーが必要です．
用意済みのイメージにはThunarとMIME関連付けを含め，保存画像はRistretto，動画はmpvで開けます．
開けなかった場合はGUI内のログに保存先とエラーを表示します．
独自のイメージでは `xdg-utils` に加え，ファイルマネージャーと
`desktop-file-utils`・`shared-mime-info` を導入してください．
共有環境での実動作確認は，ワークスペースルートから
`python3 scripts/checks/desktop_test.py --gui` で行えます．

推論・クリック座標保存・推論付き録画では，カメラSDKより先にOpenCVの画面を初期化しています．
SDKのストリーム開始後に初期化すると，環境によって画面作成やフレーム取得が停止することがあるためです．
