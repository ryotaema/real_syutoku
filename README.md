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

- `convert_bag_to_mp4.py` :bag形式をmp4形式に変換するスクリプトです．
- `d405_dataset_collect.py` :realsense D405での画像取得用スクリプトです．
- `dataset_collect.py` :realsense D435での画像取得用スクリプトです．
- `mp4_collect.py` : 録画用スクリプトです．録画ファイルはbag形式で保存されます。mp4形式に変換したい場合は、スクリプト実行後に表示される変換コマンドをコピーして実行してください．
- `yolo_detection_D435.py` :yoloで作成したモデルの推論の検証を行う用のスクリプトです．(モデルはこのリポジトリに含まれていません)

## OpenVINOのモデル変換について
私のノートPCはCPUにintel iRISxeという内蔵GPUが含まれています．
ですが，そのまま`.pt` モデルを利用することができないため
```
yolo export model=best.pt format=onnx
ovc best.onnx --output_model openvino_model/best.xml
```


## RealSence セットアップについて
このプロジェクトでは Intel RealSense SDK (librealsense) を使用しています．
セットアップ方法や対応デバイスについては公式リポジトリをご参照ください
https://github.com/IntelRealSense/librealsense

ちなみに私は以下のコマンドでpipインストールを行いました:
```
pip install pyrealsense2
```
