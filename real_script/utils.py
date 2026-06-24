import yaml
import argparse
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"

# 対応モデル名のキーワード（小文字で比較）
_CAMERA_MODELS = {
    'd435': 'D435',
    'd405': 'D405',
}


def detect_camera():
    """接続されている最初のRealSenseカメラを検出して返す。

    Returns:
        dict: {'name': str, 'model': str, 'serial': str}
              model は 'D435' / 'D405' / 'unknown' のいずれか
    Raises:
        RuntimeError: デバイスが見つからない場合
    """
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("RealSenseデバイスが接続されていません")

    dev = devices[0]
    name   = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)

    model = 'unknown'
    name_lower = name.lower()
    for key, label in _CAMERA_MODELS.items():
        if key in name_lower:
            model = label
            break

    return {'name': name, 'model': model, 'serial': serial}


def get_depth_alpha(cfg, model):
    """カメラモデルに対応する depth colormap の alpha 値を返す。
    None の場合はフレーム内相対正規化を使用する（make_depth_colormap 参照）。
    """
    return cfg['camera']['depth_alpha'].get(model, cfg['camera']['depth_alpha']['default'])


def make_depth_colormap(depth_image, alpha):
    """depth_image (uint16 numpy array) → BGR uint8 の深度カラーマップを返す。

    alpha=None  : フレーム内相対正規化（D405 向け。距離範囲に関わらず全域を使う）
    alpha=float : 固定倍率（D435 向け。絶対距離が色に対応する）
    """
    import cv2
    import numpy as np
    if alpha is None:
        valid = depth_image[depth_image > 0]
        if valid.size == 0:
            return np.zeros((*depth_image.shape, 3), dtype=np.uint8)
        normed = np.clip(
            (depth_image.astype(np.float32) - valid.min()) / (valid.max() - valid.min() + 1e-6) * 255,
            0, 255,
        ).astype(np.uint8)
        return cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET)


def load_config():
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def build_parser(include_model=False, include_conf=False, bag_input=False):
    parser = argparse.ArgumentParser()
    if bag_input:
        parser.add_argument('bag_path', help='.bagファイルのパス')
    parser.add_argument('--fps',    type=int,   default=None, metavar='N',
                        help='FPS（config.yamlの値を上書き）')
    parser.add_argument('--width',  type=int,   default=None, metavar='N',
                        help='横解像度（config.yamlの値を上書き）')
    parser.add_argument('--height', type=int,   default=None, metavar='N',
                        help='縦解像度（config.yamlの値を上書き）')
    if include_model:
        parser.add_argument('--model', type=str, default=None, metavar='PATH',
                            help='モデルパス（config.yamlの値を上書き）')
    if include_conf:
        parser.add_argument('--conf',  type=float, default=None, metavar='F',
                            help='信頼度閾値（config.yamlの値を上書き）')
    return parser


def apply_args(cfg, args, model_key='yolo_path'):
    if args.fps    is not None: cfg['camera']['fps']    = args.fps
    if args.width  is not None: cfg['camera']['width']  = args.width
    if args.height is not None: cfg['camera']['height'] = args.height
    if hasattr(args, 'model') and args.model is not None:
        cfg['model'][model_key] = args.model
    if hasattr(args, 'conf') and args.conf is not None:
        cfg['model']['confidence_threshold'] = args.conf
    return cfg
