"""カメラ取得の共通層。RealSense (pyrealsense2) と Orbbec (pyorbbecsdk) の差を吸収する。

各スクリプトは SDK を直接触らず、このモジュール経由でフレームを numpy 配列として受け取る。

    cam = open_camera(info, W, H, FPS, ir=True)   # info は utils.detect_camera() の戻り値
    cam.start()
    fr = cam.read()          # Frames | None（タイムアウト・フレーム欠落時）
    fr.color                 # BGR uint8 (H, W, 3)
    fr.depth                 # uint16 (H, W)。align=True なら color に位置合わせ済み
    fr.ir_left / ir_right    # uint8 (H, W)。位置合わせしない元の IR
    cam.stop()

open_camera(..., record_path='xxx.bag') で取得中のストリームを .bag に録画する（stop() で確定）。
.bag の形式は SDK ごとに異なり互換性は無い（RealSense / Orbbec 独自）。再生は open_playback()。

depth の単位は cam.depth_scale（m / 1単位）。D435・Gemini336 は 0.001（mm）、D405 は 0.0001。
SDK は使うときに import するので、片方しか入っていない環境でも動く。
"""
import numpy as np


class Frames:
    """1ショット分のフレーム。取得していないストリームは None。"""

    def __init__(self, color=None, depth=None, ir_left=None, ir_right=None, raw=None,
                 depth_raw=None, metadata=None):
        self.color    = color
        self.depth    = depth
        self.ir_left  = ir_left
        self.ir_right = ir_right
        self.depth_raw = depth_raw
        self.metadata = metadata or {}
        self._raw     = raw     # バックエンド固有のフレーム（点群計算用）


# ============================================================================
# デバイス列挙
# ============================================================================

def list_devices(diagnostics=None):
    """接続されている全カメラを列挙する。

    diagnostics にリストを渡すと、SDK ごとの読み込み・検出失敗の詳細を追記する。
    片方の SDK が使えない場合も、もう片方のカメラは返す。

    Returns:
        list[dict]: {'name': str, 'serial': str, 'backend': 'realsense' | 'orbbec'}
    """
    devices = []
    for label, module, enumerate_devices in (
        ('RealSense', 'pyrealsense2', _list_realsense),
        ('Orbbec', 'pyorbbecsdk', _list_orbbec),
    ):
        try:
            devices.extend(enumerate_devices())
        except Exception as exc:
            if diagnostics is not None:
                missing = isinstance(exc, ModuleNotFoundError) and exc.name == module
                reason = ('SDK がインストールされていません' if missing
                          else 'SDK の読み込み、またはカメラの検出に失敗しました')
                diagnostics.append(
                    f'{label}（{module}）: {reason}\n'
                    f'  {type(exc).__name__}: {exc}')
    return devices


def _list_realsense():
    import pyrealsense2 as rs
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
    ob = _import_orbbec()
    ctx = ob.Context()      # DeviceList より先に破棄されないよう保持する
    dl  = ctx.query_devices()
    return [{
        'name':    dl.get_device_name_by_index(i),
        'serial':  dl.get_device_serial_number_by_index(i),
        'backend': 'orbbec',
    } for i in range(dl.get_count())]


def open_camera(info, width, height, fps, color=True, depth=True, ir=False, align=True,
                record_path=None):
    """detect_camera() の戻り値からカメラを作る（start() はまだ呼ばない）。
    record_path を指定すると、有効にした全ストリームをその .bag に録画する。"""
    cls = {'realsense': _RealSenseCamera, 'orbbec': _OrbbecCamera}[info['backend']]
    return cls(info['serial'], width, height, fps, color, depth, ir, align, record_path)


# ============================================================================
# .bag 再生
# ============================================================================

def bag_backend(path):
    """.bag がどの SDK で録画されたかを返す（'realsense' / 'orbbec'）。
    どちらも ROS bag 形式なので、先頭付近のトピック名で見分ける。"""
    with open(path, 'rb') as f:
        head = f.read(1 << 20)
    if not head.startswith(b'#ROSBAG'):
        raise RuntimeError(f".bag ファイルではありません: {path}")
    if b'topic=/device_0/' in head or b'topic=/file_version' in head:
        return 'realsense'
    if b'topic=/cam/' in head:
        return 'orbbec'
    raise RuntimeError(f"録画した SDK を判別できません: {path}")


def open_playback(path):
    """.bag を color だけ再生するリーダーを作る（start() はまだ呼ばない）。
    read() は Frames（color のみ）、最後まで再生すると None を返して eof が True になる。"""
    cls = {'realsense': _RealSensePlayback, 'orbbec': _OrbbecPlayback}[bag_backend(path)]
    return cls(path)


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
    def __init__(self, serial, width, height, fps, color, depth, ir, align, record_path=None):
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.use_color, self.use_depth, self.use_ir = color, depth, ir
        self.use_align   = align and color and depth
        self.record_path = record_path
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
        # 従来の rs.points.export_to_ply（RealSense Viewer 座標系）に合わせて y・z を反転する
        write_ply(path, xyz[keep] * np.array([1, -1, -1], dtype=np.float32), rgb[keep])


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
        if self.record_path:
            cfg.enable_record_to_file(self.record_path)

        self._pipe    = rs.pipeline()
        self._profile = self._pipe.start(cfg)
        self._align   = rs.align(rs.stream.color) if self.use_align else None
        if self.use_depth:
            self.depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()

    def stop(self):
        self._pipe.stop()
        # 録画時は pipeline（内部の recorder）が破棄されたときに .bag のインデックスが書かれる
        self._pipe = self._profile = None

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
            fr.color = np.asanyarray(cf.get_data()).copy()
            fr.metadata['color'] = {'timestamp_ms': cf.get_timestamp(), 'frame_number': cf.get_frame_number()}
        if self.use_depth:
            df = aligned.get_depth_frame()
            if not df:
                return None
            fr.depth = np.asanyarray(df.get_data()).copy()
            native_depth = frames.get_depth_frame()
            fr.depth_raw = np.asanyarray(native_depth.get_data()).copy()
            fr.metadata['depth'] = {'timestamp_ms': native_depth.get_timestamp(),
                                    'frame_number': native_depth.get_frame_number()}
        if self.use_ir:
            # IR は位置合わせ前のフレームから取る（元解像度・純粋な IR 画像）
            ir1, ir2 = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not ir1 or not ir2:
                return None
            fr.ir_left  = np.asanyarray(ir1.get_data()).copy()
            fr.ir_right = np.asanyarray(ir2.get_data()).copy()
            for name, frame in [('ir_left', ir1), ('ir_right', ir2)]:
                fr.metadata[name] = {'timestamp_ms': frame.get_timestamp(), 'frame_number': frame.get_frame_number()}
        return fr

    def calibration(self):
        color_profile = self._profile.get_stream(self._rs.stream.color)
        streams = {}
        for profile in self._profile.get_streams():
            video = profile.as_video_stream_profile()
            intr = video.get_intrinsics()
            extr = profile.get_extrinsics_to(color_profile)
            name = str(profile.stream_type()) + '_' + str(profile.stream_index())
            streams[name] = {
                'width': intr.width, 'height': intr.height,
                'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
                'model': str(intr.model), 'coeffs': list(intr.coeffs),
                'to_color': {'rotation': list(extr.rotation), 'translation_m': list(extr.translation)},
            }
        return {'color': self.intrinsics(), 'streams': streams, 'depth_scale': self.depth_scale}

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
        # 録画は pipeline の開始前に RecordDevice を作っておく必要がある
        self._recorder = ob.RecordDevice(dev, self.record_path) if self.record_path else None
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
        # RecordDevice を破棄すると .bag が閉じられる（pipeline 停止より先に行う）
        self._recorder = None
        self._pipe.stop()

    @staticmethod
    def _image(frame, dtype, channels=1):
        return _orbbec_image(frame, dtype, channels)

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
            fr.metadata['color'] = {'timestamp_us': int(cf.get_timestamp_us()), 'frame_number': int(cf.get_index())}
        if self.use_depth:
            df = aligned.get_depth_frame()
            if df is None:
                return None
            fr.depth = self._image(df, np.uint16)
            native_depth = frames.get_depth_frame()
            fr.depth_raw = self._image(native_depth, np.uint16)
            fr.metadata['depth'] = {'timestamp_us': int(native_depth.get_timestamp_us()),
                                    'frame_number': int(native_depth.get_index())}
            # get_depth_scale() は mm / 1単位
            self.depth_scale = df.get_depth_scale() * 0.001
        if self.use_ir:
            l = frames.get_frame(self._ob.OBFrameType.LEFT_IR_FRAME)
            r = frames.get_frame(self._ob.OBFrameType.RIGHT_IR_FRAME)
            if l is None or r is None:
                return None
            fr.ir_left  = self._image(l, np.uint8)
            fr.ir_right = self._image(r, np.uint8)
            for name, frame in [('ir_left', l), ('ir_right', r)]:
                fr.metadata[name] = {'timestamp_us': int(frame.get_timestamp_us()), 'frame_number': int(frame.get_index())}
        return fr

    def calibration(self):
        def sensor(intr, dist):
            return {'width': intr.width, 'height': intr.height,
                    'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.cx, 'ppy': intr.cy,
                    'model': str(dist.model),
                    'coeffs': [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3, dist.k4, dist.k5, dist.k6]}
        return {
            'color': sensor(self._param.rgb_intrinsic, self._param.rgb_distortion),
            'depth': sensor(self._param.depth_intrinsic, self._param.depth_distortion),
            'depth_to_color': {'rotation': np.asarray(self._param.transform.rot, dtype=float).reshape(-1).tolist(),
                               'translation_m': (np.asarray(self._param.transform.transform, dtype=float)
                                                 .reshape(-1) * 0.001).tolist()},
            'depth_scale': self.depth_scale,
        }

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


def _orbbec_image(frame, dtype, channels=1):
    vf = frame.as_video_frame()
    shape = (vf.get_height(), vf.get_width()) + ((channels,) if channels > 1 else ())
    # SDK のバッファは次フレームで再利用されるので copy しておく
    return np.frombuffer(vf.get_data(), dtype=dtype).reshape(shape).copy()


# ============================================================================
# 再生（color のみ）
# ============================================================================

class _PlaybackBase:
    def __init__(self, path):
        self.path = path
        self.eof  = False
        self.fps  = None    # start() 後に確定
        self.size = None    # (W, H)。start() 後に確定

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class _RealSensePlayback(_PlaybackBase):

    def start(self):
        import pyrealsense2 as rs
        self._rs = rs
        cfg = rs.config()
        cfg.enable_device_from_file(self.path, repeat_playback=False)
        cfg.enable_stream(rs.stream.color)     # 解像度・形式は録画時のもの
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        vp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.fps, self.size = vp.fps(), (vp.width(), vp.height())
        self._rgb = vp.format() == rs.format.rgb8

    def stop(self):
        self._pipe.stop()

    def read(self, timeout_ms=5000):
        try:
            frames = self._pipe.wait_for_frames(timeout_ms)
        except RuntimeError:        # 再生終了後はフレームが来ずタイムアウトする
            self.eof = True
            return None
        cf = frames.get_color_frame()
        if not cf:
            return None
        color = np.asanyarray(cf.get_data())
        return Frames(color=color[..., ::-1].copy() if self._rgb else color)


class _OrbbecPlayback(_PlaybackBase):

    def start(self):
        ob = _import_orbbec()
        self._ob = ob
        self._pb   = ob.PlaybackDevice(self.path)
        self._pipe = ob.Pipeline(self._pb)
        cfg = ob.Config()
        cfg.enable_stream(ob.OBSensorType.COLOR_SENSOR)   # 解像度・形式は録画時のもの
        self._pipe.start(cfg)
        # fps / size は最初のフレームから取る（read() で返せるよう保持しておく）
        self._first = None
        while self._first is None and not self.eof:
            self._first = self.read()
        if self._first is not None:
            cf = self._first._raw
            vp = cf.get_stream_profile().as_video_stream_profile()
            self.fps, self.size = vp.get_fps(), (cf.get_width(), cf.get_height())

    def stop(self):
        self._pipe.stop()

    def read(self, timeout_ms=1000):
        if getattr(self, '_first', None) is not None:
            fr, self._first = self._first, None
            return fr
        fs = self._pipe.wait_for_frames(timeout_ms)
        if fs is None:
            if self._pb.get_playback_status() == self._ob.OBPlaybackStatus.STOPPED:
                self.eof = True
            return None
        cf = fs.get_color_frame()
        if cf is None:
            return None
        fmt = cf.get_format()
        if fmt not in (self._ob.OBFormat.BGR, self._ob.OBFormat.RGB):
            raise RuntimeError(f"未対応の color 形式です: {fmt}")
        color = _orbbec_image(cf, np.uint8, 3)
        if fmt == self._ob.OBFormat.RGB:
            color = color[..., ::-1].copy()
        return Frames(color=color, raw=cf)
