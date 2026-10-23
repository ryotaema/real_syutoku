"""カメラ取得の共通層。RealSense (pyrealsense2) と Orbbec (pyorbbecsdk) の差を吸収する。

各スクリプトは SDK を直接触らず、このモジュール経由でフレームを numpy 配列として受け取る。

    cam = open_camera(info, W, H, FPS, ir=True)   # info は utils.detect_camera() の戻り値
    cam.start()
    fr = cam.read()          # Frames | None（タイムアウト・フレーム欠落時）
    fr.color                 # BGR uint8 (H, W, 3)
    fr.depth                 # uint16 (H, W)。align=True なら color に位置合わせ済み
    fr.ir_left / ir_right    # uint8 (H, W)。位置合わせしない元の IR
    cam.stop()

depth の単位は cam.depth_scale（m / 1単位）。D435・Gemini336 は 0.001（mm）、D405 は 0.0001。
SDK は使うときに import するので、片方しか入っていない環境でも動く。
"""
import numpy as np


class Frames:
    """1ショット分のフレーム。取得していないストリームは None。"""

    def __init__(self, color=None, depth=None, ir_left=None, ir_right=None, raw=None):
        self.color    = color
        self.depth    = depth
        self.ir_left  = ir_left
        self.ir_right = ir_right
        self._raw     = raw     # バックエンド固有のフレーム（点群計算用）


# ============================================================================
# デバイス列挙
# ============================================================================

def list_devices():
    """接続されている全カメラを列挙する。

    Returns:
        list[dict]: {'name': str, 'serial': str, 'backend': 'realsense' | 'orbbec'}
    """
    return _list_realsense() + _list_orbbec()


def _list_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    devs = []
    for dev in rs.context().query_devices():
        devs.append({
            'name':    dev.get_info(rs.camera_info.name),
            'serial':  dev.get_info(rs.camera_info.serial_number),
            'backend': 'realsense',
        })
    return devs


def _import_orbbec():
    import pyorbbecsdk as ob
    # 既定ではカレントディレクトリに Log/ を作るので抑止する
    ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
    ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
    return ob


def _list_orbbec():
    try:
        ob = _import_orbbec()
    except ImportError:
        return []
    ctx = ob.Context()      # DeviceList より先に破棄されないよう保持する
    dl  = ctx.query_devices()
    return [{
        'name':    dl.get_device_name_by_index(i),
        'serial':  dl.get_device_serial_number_by_index(i),
        'backend': 'orbbec',
    } for i in range(dl.get_count())]


def open_camera(info, width, height, fps, color=True, depth=True, ir=False, align=True):
    """detect_camera() の戻り値からカメラを作る（start() はまだ呼ばない）。"""
    cls = {'realsense': _RealSenseCamera, 'orbbec': _OrbbecCamera}[info['backend']]
    return cls(info['serial'], width, height, fps, color, depth, ir, align)


# ============================================================================
# 点群
# ============================================================================

def write_ply(path, xyz, rgb):
    """xyz (N,3) float [m] と rgb (N,3) uint8 をバイナリ PLY で書き出す。"""
    n = len(xyz)
    vert = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vert['x'], vert['y'], vert['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vert['red'], vert['green'], vert['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vert.tobytes())


class _CameraBase:
    def __init__(self, serial, width, height, fps, color, depth, ir, align):
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.use_color, self.use_depth, self.use_ir = color, depth, ir
        self.use_align   = align and color and depth
        self.depth_scale = None     # m / 1単位。start() 後に確定

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def save_ply(self, fr, path, min_depth=None, max_depth=None):
        """color に位置合わせ済みの depth から色付き点群を作って PLY に保存する。
        min_depth / max_depth [m] の範囲外と depth=0 の点は捨てる。"""
        xyz = self._vertices(fr).reshape(-1, 3)
        rgb = fr.color[..., ::-1].reshape(-1, 3)   # BGR → RGB
        z = xyz[:, 2]
        keep = z > 0
        if min_depth is not None:
            keep &= z >= min_depth
        if max_depth is not None:
            keep &= z <= max_depth
        write_ply(path, xyz[keep], rgb[keep])


# ============================================================================
# RealSense
# ============================================================================

class _RealSenseCamera(_CameraBase):

    def start(self):
        import pyrealsense2 as rs
        self._rs = rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        W, H, FPS = self.width, self.height, self.fps
        try:
            if self.use_color:
                cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            if self.use_depth:
                cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            if self.use_ir:
                cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
                cfg.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
        except RuntimeError as e:
            raise RuntimeError(f"ストリームの設定に失敗しました: {e}")

        self._pipe    = rs.pipeline()
        self._profile = self._pipe.start(cfg)
        self._align   = rs.align(rs.stream.color) if self.use_align else None
        if self.use_depth:
            self.depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()

    def stop(self):
        self._pipe.stop()

    def read(self, timeout_ms=5000):
        try:
            frames = self._pipe.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None
        aligned = self._align.process(frames) if self._align else frames

        fr = Frames(raw=aligned)
        if self.use_color:
            cf = aligned.get_color_frame()
            if not cf:
                return None
            fr.color = np.asanyarray(cf.get_data())
        if self.use_depth:
            df = aligned.get_depth_frame()
            if not df:
                return None
            fr.depth = np.asanyarray(df.get_data())
        if self.use_ir:
            # IR は位置合わせ前のフレームから取る（元解像度・純粋な IR 画像）
            ir1, ir2 = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not ir1 or not ir2:
                return None
            fr.ir_left  = np.asanyarray(ir1.get_data())
            fr.ir_right = np.asanyarray(ir2.get_data())
        return fr

    def intrinsics(self):
        intr = (self._profile.get_stream(self._rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        return {
            'width':  intr.width,
            'height': intr.height,
            'fx':     intr.fx,
            'fy':     intr.fy,
            'ppx':    intr.ppx,
            'ppy':    intr.ppy,
            'model':  str(intr.model),
            'coeffs': list(intr.coeffs),
        }

    def _vertices(self, fr):
        pts = self._rs.pointcloud().calculate(fr._raw.get_depth_frame())
        return np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)


# ============================================================================
# Orbbec（Gemini 336 で確認）
# ============================================================================

class _OrbbecCamera(_CameraBase):

    def start(self):
        ob = _import_orbbec()
        self._ob = ob
        self._ctx = ob.Context()   # Device より先に破棄されないよう保持する
        dev = self._ctx.query_devices().get_device_by_serial_number(self.serial)
        self._pipe = ob.Pipeline(dev)
        cfg = ob.Config()
        W, H, FPS = self.width, self.height, self.fps
        S = ob.OBSensorType
        try:
            if self.use_color:
                cfg.enable_video_stream(S.COLOR_SENSOR, W, H, FPS, ob.OBFormat.BGR)
            if self.use_depth:
                cfg.enable_video_stream(S.DEPTH_SENSOR, W, H, FPS, ob.OBFormat.Y16)
            if self.use_ir:
                cfg.enable_video_stream(S.LEFT_IR_SENSOR,  W, H, FPS, ob.OBFormat.Y8)
                cfg.enable_video_stream(S.RIGHT_IR_SENSOR, W, H, FPS, ob.OBFormat.Y8)
            # 有効にした全ストリームが揃ったフレームセットだけを受け取る
            cfg.set_frame_aggregate_output_mode(
                ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self._pipe.start(cfg)
        except Exception as e:
            raise RuntimeError(f"ストリームの設定に失敗しました（{W}x{H}@{FPS}）: {e}")

        self._align = (ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
                       if self.use_align else None)
        self._param = self._pipe.get_camera_param()

    def stop(self):
        self._pipe.stop()

    @staticmethod
    def _image(frame, dtype, channels=1):
        vf = frame.as_video_frame()
        shape = (vf.get_height(), vf.get_width()) + ((channels,) if channels > 1 else ())
        # SDK のバッファは次フレームで再利用されるので copy しておく
        return np.frombuffer(vf.get_data(), dtype=dtype).reshape(shape).copy()

    def read(self, timeout_ms=5000):
        frames = self._pipe.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        aligned = frames
        if self._align:
            res = self._align.process(frames)
            if res is None:
                return None
            aligned = res.as_frame_set()

        fr = Frames(raw=aligned)
        if self.use_color:
            cf = aligned.get_color_frame()
            if cf is None:
                return None
            fr.color = self._image(cf, np.uint8, 3)
        if self.use_depth:
            df = aligned.get_depth_frame()
            if df is None:
                return None
            fr.depth = self._image(df, np.uint16)
            # get_depth_scale() は mm / 1単位
            self.depth_scale = df.get_depth_scale() * 0.001
        if self.use_ir:
            l = frames.get_frame(self._ob.OBFrameType.LEFT_IR_FRAME)
            r = frames.get_frame(self._ob.OBFrameType.RIGHT_IR_FRAME)
            if l is None or r is None:
                return None
            fr.ir_left  = self._image(l, np.uint8)
            fr.ir_right = self._image(r, np.uint8)
        return fr

    def intrinsics(self):
        i, d = self._param.rgb_intrinsic, self._param.rgb_distortion
        return {
            'width':  i.width,
            'height': i.height,
            'fx':     i.fx,
            'fy':     i.fy,
            'ppx':    i.cx,
            'ppy':    i.cy,
            'model':  str(d.model),
            'coeffs': [d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6],
        }

    def _vertices(self, fr):
        # color に位置合わせ済みの depth を color の内部パラメータで逆投影する
        i = self._param.rgb_intrinsic
        h, w = fr.depth.shape
        z = fr.depth.astype(np.float32) * self.depth_scale
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (u - i.cx) * z / i.fx
        y = (v - i.cy) * z / i.fy
        return np.stack((x, y, z), axis=-1)
