import yaml
import argparse
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"


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
