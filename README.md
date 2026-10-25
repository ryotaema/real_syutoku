# real_syutoku

Intel RealSense（D435 / D405）・Orbbec（Gemini 336）カメラを用いた画像取得・アノテーション・推論モデル検証用スクリプト群です．
撮影，録画，タイムラプス，点群マージ，YOLO 検出，アノテーション補助を GUI からボタンで実行できます．
接続されたカメラは自動で認識します．

## セットアップ

ROS は不要です．Python とメーカー SDK でカメラへ直接接続します．リポジトリのルートで次を実行します．

```bash
pip install -r real_script/requirements.txt   # pyrealsense2 / pyorbbecsdk2 は使う方だけでよい
sudo apt install python3-tk                    # GUI・アノテーションツール用

# 点群の合わせ込み（point_merge.py）を使う場合のみ
pip install -r real_script/requirements_processing.txt
```

Orbbec を初めて使う PC では udev ルールの設定が必要です．
SDK の詳細・`config.yaml` / `config.local.yaml`・Docker で使う場合は [docs/setup.md](docs/setup.md) を参照してください．

## 使い方

```bash
python3 real_script/gui.py
```

1. 「カメラ」に接続中のカメラが表示されることを確認します（2台以上つないでいるときはここで選びます）．
2. タブ（画像・点群・動画・タイムラプス・推論・アノテーション）を開いてボタンを押します．
3. 開いたウィンドウで，ボタン横の「操作」に書かれたキーで保存・終了します．途中でやめるときは「停止」．

保存先は `data/` 配下です（「保存先を開く」で開けます）．
各タブ・ボタンの説明と困ったときの対処は **[docs/gui.md](docs/gui.md)**（GUI の「使い方」ボタンからも開けます）を参照してください．各スクリプトはコマンドラインからも実行できます．

## ドキュメント

| 資料 | 内容 |
|---|---|
| [docs/gui.md](docs/gui.md) | GUI の使い方（画面の見方，タブ・ボタンごとの機能，困ったとき） |
| [docs/setup.md](docs/setup.md) | セットアップ詳細（RealSense / Orbbec SDK，`config.yaml` / `config.local.yaml`，Docker） |
| [docs/usage.md](docs/usage.md) | GUI の詳細，コマンドラインでの使い方，CLI オプション一覧，OpenVINO |
| [docs/data_format.md](docs/data_format.md) | カメラ自動認識と機種ごとの違い，ファイルの命名規則，保存される形式 |
| [docs/legacy_conversion.md](docs/legacy_conversion.md) | 旧命名データの変換 |

## ライセンス

このリポジトリは [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) のもとで公開されています。

本プロジェクトは [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）を使用しているため、AGPL-3.0 に従い同ライセンスを適用しています。
