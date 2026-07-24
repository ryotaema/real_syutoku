import yaml
import argparse
import json
import os
from datetime import datetime
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
    parser.add_argument('--tag',    type=str,   default=None, metavar='NAME',
                        help='セッションディレクトリ名に付ける任意タグ 例: greenhouse'
                             '（ファイル名には付かない）')
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


# ============================================================================
# データ命名規則  （real_syutoku / nyx660_syutoku 共通。両者で同一の実装を置く）
#
#   {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.{ext}
#   例) d435_260707_101741_00042_c.jpg
#
#   cam    : カメラコード（d435 / d405 / nyx）。リポジトリを跨いでも衝突しない
#   YYMMDD : 取得日
#   HHMMSS : セッション開始時刻。1プロセス＝1セッションで固定値
#   NNNNN  : セッション内のショット連番（撮影時刻ではない）。
#            同一ショットの color/depth/ir は全て同じ番号を共有するため、
#            末尾トークンの差し替えだけで対応するファイルを引ける
#   mod    : モダリティコード（下表 MOD_CODES）
#
# ファイル名だけで「どのカメラの・いつの・何枚目の・何の画像か」が復元できるので、
# 学習用に flatten しても labelImg にまとめて放り込んでも識別性が壊れない。
# 撮影条件などファイル名に載せない情報は metadata.json に記録する。
# ============================================================================

# サブディレクトリ名 → モダリティコード
# （キーがそのままセッション配下のサブディレクトリ名になる）
MOD_CODES = {
    'color':          'c',
    'depth':          'd',      # 16bit raw（png）
    'depth_colormap': 'dc',
    'ir':             'i1',     # 単眼IR（NYX660）
    'ir_left':        'i1',
    'ir_right':       'i2',
    'ir_left_color':  'i1c',
    'ir_right_color': 'i2c',
    'pointcloud':     'pc',
    'points':         'pt',     # クリック座標などのテキスト
    'detected':       'det',    # YOLO描画済み
    'annotated':      'det',
}


def cam_code(model):
    """カメラモデル名 → ファイル名用のカメラコード。'D435' → 'd435'"""
    return model.lower() if model in _CAMERA_MODELS.values() else 'unk'


def mod_code(modality):
    """モダリティ名 → コード。未登録の名前はそのまま使う。"""
    return MOD_CODES.get(modality, modality)


def make_prefix(cam, started_at=None):
    """セッションprefix（{cam}_{YYMMDD}_{HHMMSS}）を返す。mp4/bag など
    セッションディレクトリを作らない出力でも同じ規則を使うためのヘルパー。"""
    now = started_at or datetime.now()
    return f"{cam}_{now.strftime('%y%m%d')}_{now.strftime('%H%M%S')}"


class Session:
    """1回の取得セッション。保存先ディレクトリとファイル名の生成を一元管理する。

    <base_dir>/<YYMMDD>/<prefix>[_<tag>]/<modality>/<prefix>_<NNNNN>_<mod>.<ext>

    tag はセッションディレクトリ名にだけ付く（ファイル名の桁を汚さないため）。
    ディレクトリ名はファイル名の prefix に前方一致するので対応関係は保たれる。
    """

    def __init__(self, base_dir, cam, tag=None, subdirs=(), started_at=None):
        self.started_at = started_at or datetime.now()
        self.cam    = cam
        self.tag    = tag
        self.date   = self.started_at.strftime('%y%m%d')
        self.time   = self.started_at.strftime('%H%M%S')
        self.prefix = make_prefix(cam, self.started_at)

        dir_name = f"{self.prefix}_{tag}" if tag else self.prefix
        self.dir = Path(os.path.expanduser(str(base_dir))) / self.date / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)
        for sub in subdirs:
            (self.dir / sub).mkdir(exist_ok=True)

    def name(self, idx, modality, ext='jpg'):
        """ファイル名のみを返す。"""
        return f"{self.prefix}_{idx:05d}_{mod_code(modality)}.{ext}"

    def path(self, idx, modality, ext='jpg', sub=True):
        """保存先のフルパスを返す（ディレクトリは自動作成）。

        sub=True  : <dir>/<modality>/ に置く（既定）
        sub=False : <dir>/ 直下に置く
        sub='xxx' : <dir>/xxx/ に置く
        """
        d = self.dir if sub is False else self.dir / (modality if sub is True else sub)
        d.mkdir(parents=True, exist_ok=True)
        return str(d / self.name(idx, modality, ext))

    def file(self, suffix, ext):
        """連番を持たないセッション単位のファイル（動画など）のフルパスを返す。"""
        return str(self.dir / f"{self.prefix}_{suffix}.{ext}")

    def write_metadata(self, **info):
        """metadata.json を書き出す。撮影条件など、ファイル名に載せない情報はここに。"""
        meta = {
            'session_id': self.prefix,
            'camera_code': self.cam,
            'started_at': self.started_at.isoformat(timespec='seconds'),
            'tag': self.tag,
            'naming': '{cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.{ext}',
        }
        meta.update(info)
        with open(self.dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
