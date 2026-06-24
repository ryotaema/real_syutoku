"""接続されているRealSenseカメラを自動検出し、対応するデータ収集スクリプトを起動する。

対応スクリプト:
  D435 → dataset_collect.py       (Enter で保存開始、color/depth/IR)
  D405 → d405_dataset_collect.py  (連続30FPSで保存、color/depth/IR)

未対応のカメラが接続されている場合は D435 スクリプトにフォールバックする。
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import detect_camera

_SCRIPT_DIR = Path(__file__).parent

_MODEL_TO_SCRIPT = {
    'D435': _SCRIPT_DIR / 'dataset_collect.py',
    'D405': _SCRIPT_DIR / 'd405_dataset_collect.py',
}
_FALLBACK_SCRIPT = _SCRIPT_DIR / 'dataset_collect.py'


def main():
    try:
        cam = detect_camera()
    except RuntimeError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    print(f"検出されたカメラ: {cam['name']}  (シリアル: {cam['serial']})")

    script = _MODEL_TO_SCRIPT.get(cam['model'], _FALLBACK_SCRIPT)
    if cam['model'] == 'unknown':
        print(f"未対応のモデルです。フォールバック: {script.name}")
    else:
        print(f"起動スクリプト: {script.name}")

    # 自分に渡された引数をそのままスクリプトへ転送（--fps, --width, --height など）
    extra_args = sys.argv[1:]
    cmd = [sys.executable, str(script)] + extra_args
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
