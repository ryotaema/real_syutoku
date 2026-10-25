"""1ショットの画像と任意のPLYを、同じ番号で保存する。ROSは不要。"""
from datetime import datetime
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from utils import Session


def create_session(base_dir, cam, tag=None):
    # 同じ秒に再撮影しても既存セッションへ上書きしない。
    while True:
        now = datetime.now()
        prefix = f"{cam}_{now:%y%m%d_%H%M%S}"
        directory = Path(base_dir) / now.strftime('%y%m%d') / (f'{prefix}_{tag}' if tag else prefix)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            time.sleep(0.02)
            continue
        return Session(base_dir, cam=cam, tag=tag, started_at=now)


def write_bundle(session, index, images, pointcloud_writer=None):
    """書き込み失敗を検出し、指定した画像・任意のPLYが揃ってから保存先へ移す。"""
    targets = {name: Path(session.path(index, name, ext=ext))
               for name, (_, ext) in images.items()}
    if pointcloud_writer is not None:
        targets['pointcloud'] = Path(session.path(index, 'pointcloud', ext='ply'))
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f'既存データは上書きしません: {path}')
    published = []
    with tempfile.TemporaryDirectory(prefix='.shot-', dir=session.dir) as directory:
        staged = {}
        for name, (image, _) in images.items():
            path = Path(directory) / targets[name].name
            if not cv2.imwrite(str(path), image):
                raise OSError(f'画像を書き込めません: {targets[name]}')
            staged[name] = path
        if pointcloud_writer is not None:
            path = Path(directory) / targets['pointcloud'].name
            pointcloud_writer(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise OSError('PLYを書き込めません')
            staged['pointcloud'] = path
        try:
            for name, path in staged.items():
                os.replace(path, targets[name])
                published.append(targets[name])
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return {name: str(path.relative_to(session.dir)) for name, path in targets.items()}


def preview_grid(color, depth_colormap, infrared):
    height, width = depth_colormap.shape[:2]
    tiles = [cv2.resize(color, (width, height)), depth_colormap]
    for image in infrared:
        if image is not None:
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tiles.append(cv2.applyColorMap(cv2.resize(image, (width, height)), cv2.COLORMAP_JET))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(depth_colormap))
    return np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])


def save_frame_bundle(session, index, camera, frame, depth_colormap, all_data=False):
    images = {'color': (frame.color, 'jpg'), 'depth_colormap': (depth_colormap, 'jpg')}
    if all_data:
        images.update(color_raw=(frame.color, 'png'), depth=(frame.depth, 'png'))
        if frame.depth_raw is not None:
            images['depth_raw'] = (frame.depth_raw, 'png')
    for name in ('ir_left', 'ir_right'):
        image = getattr(frame, name)
        if image is not None:
            images[name] = (image, 'png' if all_data else 'jpg')
            images[name + '_color'] = (cv2.applyColorMap(image, cv2.COLORMAP_JET), 'jpg')
    writer = (lambda path: camera.save_ply(frame, path)) if all_data else None
    files = write_bundle(session, index, images, writer)
    return {'index': index, 'files': files, 'streams': frame.metadata,
            'shapes': {name: list(image.shape) for name, (image, _) in images.items()}}
