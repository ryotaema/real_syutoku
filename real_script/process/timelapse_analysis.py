"""
timelapse_detect.py が出力した detection_log.csv を読み込み、
認識率（検出数・信頼度）の時系列グラフを生成する。

使い方:
  python3 process/timelapse_analysis.py <session_dir>
  python3 process/timelapse_analysis.py   # 最新セッションを自動選択

出力: <session_dir>/analysis.png
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("matplotlib が必要です: pip install matplotlib")
    sys.exit(1)


def _find_latest_session() -> Path:
    base = Path.home() / 'annot_labelimg' / 'real_syutoku' / 'data' / 'timelapse'
    sessions = sorted(base.glob('*/detection_log.csv'))
    if not sessions:
        print(f"セッションが見つかりません: {base}")
        sys.exit(1)
    return sessions[-1].parent


def _load_log(log_path: Path):
    timestamps, elapsed, n_det, avg_conf = [], [], [], []
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                timestamps.append(datetime.fromisoformat(row['timestamp']))
                elapsed.append(float(row['elapsed_min']))
                n_det.append(int(row['num_detections']))
                avg_conf.append(float(row['avg_conf']))
            except (ValueError, KeyError):
                continue
    return timestamps, elapsed, n_det, avg_conf


def main():
    if len(sys.argv) >= 2:
        session_dir = Path(sys.argv[1])
    else:
        session_dir = _find_latest_session()

    log_path = session_dir / 'detection_log.csv'
    if not log_path.exists():
        print(f"ログが見つかりません: {log_path}")
        sys.exit(1)

    print(f"ログ読み込み: {log_path}")
    timestamps, elapsed, n_det, avg_conf = _load_log(log_path)

    if not timestamps:
        print("データが空です。")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Timelapse Detection Log  —  {session_dir.name}", fontsize=12)

    ax1.plot(elapsed, n_det, marker='o', markersize=3, linewidth=1, color='steelblue')
    ax1.set_ylabel('検出数 (個)')
    ax1.set_title('検出数の時系列変化')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(elapsed, avg_conf, marker='o', markersize=3, linewidth=1, color='darkorange')
    ax2.set_ylabel('平均信頼度')
    ax2.set_xlabel('経過時間 (分)')
    ax2.set_title('平均信頼度の時系列変化')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out_path = session_dir / 'analysis.png'
    fig.savefig(out_path, dpi=150)
    print(f"グラフ保存: {out_path}")

    total = len(n_det)
    detected = sum(1 for n in n_det if n > 0)
    print(f"\n--- サマリー ---")
    print(f"総撮影数  : {total} 枚")
    print(f"検出あり  : {detected} 枚 ({100*detected/total:.1f}%)")
    print(f"平均検出数: {sum(n_det)/total:.2f} 個/枚")
    print(f"平均信頼度: {sum(avg_conf)/total:.3f}")


if __name__ == '__main__':
    main()
