# real_syutoku

このリポジトリは、ロボットの画像取得・アノテーション用データ収集・推論modelの検証用スクリプト郡．

## 構成

- `data/`：画像保存用ディレクトリ
- `scripts/`：カメラ操作・保存スクリプト

## 使用方法

1. Realsense カメラを接続
2. スクリプトを実行

```
#利用例(画像の取得)
python3 real_script/dataset_collect.py
```
## 各スクリプト

- `convert_bag_to_mp4.py` :bag形式をmp4形式に変換するスクリプト
- `d405_dataset_collect.py` :realsense D405での画像取得用スクリプト
- `dataset_collect.py` :realsense D435での画像取得用スクリプト
- `mp4_collect.py` : 録画を行うためのスクリプト(出力はbag形式のためmp4に変換してください)
- `yolo_detection_D435.py` :yoloで作成したモデルの推論の検証を行う用のスクリプト(モデルはこのリポジトリに含まれていません)

## RealSence セットアップについて
このプロジェクトでは Intel RealSense SDK (librealsense) を使用しています．
セットアップ方法や対応デバイスについては公式リポジトリをご参照ください
https://github.com/IntelRealSense/librealsense

ちなみに私は以下のコマンドでpipインストールを行いました:
```
pip install pyrealsense2
```
