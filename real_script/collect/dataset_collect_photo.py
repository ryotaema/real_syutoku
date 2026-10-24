"""RGB・深度・IR・PLYを、[s] または --once で同時保存する。"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (load_config, build_parser, apply_args, detect_camera, get_depth_alpha,
                   make_depth_colormap, cam_code, camera_metadata)
from camera import open_camera
from snapshot import create_session, save_frame_bundle, preview_grid


def main():
    parser = build_parser()
    parser.add_argument('--once', action='store_true', help='ウォームアップ後、全データを1枚保存して終了')
    parser.add_argument('--warmup', type=float, default=2.0, help='露光安定待ち時間（秒、既定2）')
    parser.add_argument('--timeout', type=float, default=10.0, help='必要なフレームの取得待ち上限（秒）')
    parser.add_argument('--no-preview', action='store_true', help='--once時に画面を表示しない')
    args = parser.parse_args()
    if args.warmup < 0 or args.timeout <= 0 or (args.no_preview and not args.once):
        parser.error('warmupは0以上、timeoutは正の値、no-previewはonceと併用してください')
    cfg = apply_args(load_config(), args)
    window = 'RealSense'
    camera = session = None
    started = False
    frames_meta = []
    metadata = {'shot_count': 0, 'frames': frames_meta}
    error = None
    try:
        if not args.no_preview:
            # SDKより先にHighGUIを初期化する（Qtとストリーム開始順の競合を避ける）。
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)
        info = detect_camera(args.serial)
        has_ir = info['model'] != 'D405'
        width, height, fps = (cfg['camera'][key] for key in ('width', 'height', 'fps'))
        alpha = get_depth_alpha(cfg, info['model'])
        camera = open_camera(info, width, height, fps, ir=has_ir)
        camera.start()
        started = True
        session = create_session(cfg['output']['images_dir'], cam_code(info['model']), args.tag)
        print(f'保存先: {session.dir}', flush=True)
        print('RGB・深度・IR・PLYを保存します。[s] 保存  [q] 終了', flush=True)
        begin = last_frame = time.monotonic()
        while True:
            frame = camera.read(timeout_ms=1000)
            key = -1
            if frame is None:
                if not args.no_preview and cv2.waitKey(1) & 0xff == ord('q'):
                    break
                if time.monotonic() - last_frame > args.timeout:
                    raise RuntimeError('RGB・深度・対応IRのフレームを取得できません。接続と設定を確認してください。')
                continue
            last_frame = time.monotonic()
            depth_cm = make_depth_colormap(frame.depth, alpha)
            if not args.no_preview:
                preview = preview_grid(frame.color, depth_cm, [frame.ir_left, frame.ir_right])
                cv2.putText(preview, f'[s] Save all  [q] Quit   Saved: {len(frames_meta)}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
            if time.monotonic() - begin < args.warmup:
                continue
            if args.once or key == ord('s'):
                if not frames_meta:
                    (session.dir / 'intrinsics.json').write_text(json.dumps(camera.intrinsics(), indent=2))
                    (session.dir / 'calibration.json').write_text(json.dumps(camera.calibration(), indent=2))
                    (session.dir / 'camera_settings.json').write_text(json.dumps(cfg['camera'], indent=2))
                record = save_frame_bundle(session, len(frames_meta) + 1, camera, frame, depth_cm)
                record['saved_at'] = datetime.now(timezone.utc).isoformat()
                frames_meta.append(record)
                metadata.update(camera=camera_metadata(info, width, height, fps, camera),
                                modalities=list(record['files']), shot_count=len(frames_meta), frames=frames_meta,
                                ply_format={'unit': 'm', 'axes': 'realsense_viewer', 'color': True},
                                depth_aligned_to='color', depth_raw_aligned=False,
                                unavailable_modalities=[] if has_ir else ['ir_left', 'ir_right'])
                session.write_metadata(**metadata)
                print(f'[{len(frames_meta)}枚目] 全データ保存完了（PLY含む）', flush=True)
                if args.once:
                    break
    except Exception as exc:  # CLIの境界で失敗を記録し、カメラはfinallyで解放する。
        error = f'{type(exc).__name__}: {exc}'
        print(f'エラー: {exc}', file=sys.stderr, flush=True)
    finally:
        try:
            if session is not None:
                session.write_metadata(**metadata, error=error)
        finally:
            try:
                if started:
                    camera.stop()
            finally:
                if not args.no_preview:
                    cv2.destroyAllWindows()
    return 1 if error else 0


if __name__ == '__main__':
    sys.exit(main())
