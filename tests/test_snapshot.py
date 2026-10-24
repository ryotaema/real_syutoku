import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / 'real_script'
sys.path.insert(0, str(SCRIPT_ROOT))
import snapshot
from utils import Session

spec = importlib.util.spec_from_file_location('photo_capture', SCRIPT_ROOT / 'collect/dataset_collect_photo.py')
photo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(photo)


class BundleIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Session(self.temp.name, cam='test')
        self.image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    def test_lossless_images_and_same_shot_number(self):
        def write_ply(path):
            Path(path).write_bytes(b'ply\nformat ascii 1.0\nend_header\n')
        depth = np.array([[0, 1], [1000, 65535]], np.uint16)
        files = snapshot.write_bundle(self.session, 7,
                                      {'color_raw': (self.image, 'png'), 'depth': (depth, 'png')}, write_ply)
        for name, expected in [('color_raw', self.image), ('depth', depth)]:
            path = self.session.dir / files[name]
            self.assertIn('_00007_', path.name)
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), expected)
        self.assertIn('_00007_pc.ply', files['pointcloud'])

    def test_image_failure_leaves_no_partial_shot(self):
        with patch.object(snapshot.cv2, 'imwrite', return_value=False):
            with self.assertRaises(OSError):
                snapshot.write_bundle(self.session, 1, {'color': (self.image, 'jpg')}, lambda path: None)
        self.assertFalse(any(p.is_file() for p in self.session.dir.rglob('*')))
        self.assertFalse(list(self.session.dir.glob('.shot-*')))

    def test_ply_failure_leaves_no_partial_shot(self):
        def fail(path):
            raise RuntimeError('PLY conversion failed')
        with self.assertRaises(RuntimeError):
            snapshot.write_bundle(self.session, 1, {'color': (self.image, 'jpg')}, fail)
        self.assertFalse(any(p.is_file() for p in self.session.dir.rglob('*')))

    def test_existing_shot_is_not_overwritten(self):
        old = Path(self.session.path(1, 'color'))
        old.write_bytes(b'original capture')
        with self.assertRaises(FileExistsError):
            snapshot.write_bundle(self.session, 1, {'color': (self.image, 'jpg')}, lambda path: None)
        self.assertEqual(old.read_bytes(), b'original capture')

    def test_sessions_do_not_share_directory(self):
        from datetime import datetime, timedelta
        now = datetime(2026, 1, 1, 0, 0, 0)
        with patch.object(snapshot, 'datetime') as clock, patch.object(snapshot.time, 'sleep'):
            clock.now.side_effect = [now, now, now + timedelta(seconds=1)]
            a = snapshot.create_session(self.temp.name, 'test')
            b = snapshot.create_session(self.temp.name, 'test')
        self.assertNotEqual(a.dir, b.dir)

    def test_preview_keeps_source_images_unmodified(self):
        original = self.image.copy()
        preview = snapshot.preview_grid(self.image, self.image, [np.ones((3, 4), np.uint8)])
        preview[:] = 0
        np.testing.assert_array_equal(self.image, original)


class FakeCamera:
    def __init__(self, infrared=True):
        from camera import Frames
        self.color = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.frame = Frames(color=self.color, depth=np.array([[1000, 0], [2000, 3000]], np.uint16),
                            depth_raw=np.arange(6, dtype=np.uint16).reshape(2, 3),
                            ir_left=np.full((2, 2), 20, np.uint8) if infrared else None,
                            ir_right=np.full((2, 2), 30, np.uint8) if infrared else None,
                            metadata={'color': {'timestamp_ms': 123.5, 'frame_number': 42}})
        self.depth_scale = 0.001 if infrared else 0.0001
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def read(self, **kwargs):
        return self.frame

    def intrinsics(self):
        return {'fx': 100, 'fy': 100, 'ppx': 1, 'ppy': 1}

    def calibration(self):
        return {'color': self.intrinsics(), 'depth_scale': self.depth_scale}

    def save_ply(self, frame, path):
        from camera import _CameraBase
        self._vertices = lambda frame: np.array([[[1, 2, 1], [0, 0, 0]], [[2, 3, 2], [3, 4, 3]]], np.float32)
        _CameraBase.save_ply(self, frame, path)


class CalibrationTests(unittest.TestCase):
    def test_orbbec_numpy_calibration_is_json_serializable_and_in_meters(self):
        from types import SimpleNamespace
        from camera import _OrbbecCamera
        camera = object.__new__(_OrbbecCamera)
        intrinsic = SimpleNamespace(width=640, height=480, fx=100., fy=100., cx=320., cy=240.)
        distortion = SimpleNamespace(model='test', **{key: 0. for key in ['k1','k2','k3','k4','k5','k6','p1','p2']})
        camera._param = SimpleNamespace(rgb_intrinsic=intrinsic, depth_intrinsic=intrinsic,
                                       rgb_distortion=distortion, depth_distortion=distortion,
                                       transform=SimpleNamespace(rot=np.eye(3, dtype=np.float32),
                                                                 transform=np.array([[10],[20],[30]],np.float32)))
        camera.depth_scale = .001
        calibration = json.loads(json.dumps(camera.calibration()))
        self.assertEqual(len(calibration['depth_to_color']['rotation']), 9)
        np.testing.assert_allclose(calibration['depth_to_color']['translation_m'], [.01,.02,.03])


class CaptureTests(unittest.TestCase):
    def run_capture(self, camera, model='G336', failed_write=False, once=True):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        cfg = {'camera': {'width': 2, 'height': 2, 'fps': 30,
                          'depth_alpha': {'G336': .4, 'D405': None, 'default': .4}},
               'output': {'images_dir': temp.name}, 'model': {}}
        info = {'name': model, 'model': model, 'serial': 'TEST', 'backend': 'orbbec'}
        with patch.object(photo, 'load_config', return_value=cfg), \
             patch.object(photo, 'detect_camera', return_value=info), \
             patch.object(photo, 'open_camera', return_value=camera) as opener, \
             patch.object(sys, 'argv', ['capture', '--warmup', '0'] + (['--once', '--no-preview'] if once else [])):
            if failed_write:
                with patch.object(snapshot.cv2, 'imwrite', return_value=False):
                    status = photo.main()
            else:
                status = photo.main()
        metadata_path = next(Path(temp.name).rglob('metadata.json'))
        return status, json.loads(metadata_path.read_text()), metadata_path.parent, opener

    def test_one_button_saves_all_data(self):
        camera = FakeCamera()
        status, metadata, directory, _ = self.run_capture(camera)
        self.assertEqual(status, 0)
        self.assertTrue(camera.stopped)
        self.assertEqual(metadata['shot_count'], 1)
        files = metadata['frames'][0]['files']
        self.assertEqual(set(files), {'color', 'color_raw', 'depth', 'depth_raw', 'depth_colormap',
                                     'ir_left', 'ir_right', 'ir_left_color', 'ir_right_color', 'pointcloud'})
        for name, image in [('color_raw', camera.color), ('depth', camera.frame.depth),
                            ('depth_raw', camera.frame.depth_raw), ('ir_left', camera.frame.ir_left)]:
            np.testing.assert_array_equal(cv2.imread(str(directory / files[name]), cv2.IMREAD_UNCHANGED), image)
        for name in ['intrinsics.json', 'calibration.json', 'camera_settings.json']:
            self.assertTrue((directory / name).is_file())
        data = (directory / files['pointcloud']).read_bytes().split(b'end_header\n')[1]
        vertices = np.frombuffer(data, dtype=[('x','<f4'), ('y','<f4'), ('z','<f4'), ('r','u1'), ('g','u1'), ('b','u1')])
        np.testing.assert_allclose(vertices['z'], [-1, -2, -3])
        np.testing.assert_allclose(vertices['y'], [-2, -3, -4])
        self.assertEqual(metadata['frames'][0]['streams']['color']['frame_number'], 42)

    def test_d405_saves_without_unsupported_ir(self):
        camera = FakeCamera(infrared=False)
        status, metadata, _, opener = self.run_capture(camera, model='D405')
        self.assertEqual(status, 0)
        self.assertFalse(opener.call_args.kwargs['ir'])
        self.assertNotIn('ir_left', metadata['frames'][0]['files'])
        self.assertEqual(metadata['unavailable_modalities'], ['ir_left', 'ir_right'])
        self.assertEqual(metadata['camera']['depth_scale'], .0001)

    def test_failed_save_is_not_counted_and_camera_is_closed(self):
        camera = FakeCamera()
        status, metadata, directory, _ = self.run_capture(camera, failed_write=True)
        self.assertEqual(status, 1)
        self.assertEqual(metadata['shot_count'], 0)
        self.assertTrue(metadata['error'])
        self.assertTrue(camera.stopped)
        self.assertFalse(list(directory.rglob('*.ply')))
        self.assertFalse(list(directory.rglob('*.jpg')))


    def test_manual_save_key_captures_one_complete_shot(self):
        camera = FakeCamera()
        with patch.object(photo.cv2, 'namedWindow'), patch.object(photo.cv2, 'imshow'), \
             patch.object(photo.cv2, 'destroyAllWindows'), \
             patch.object(photo.cv2, 'waitKey', side_effect=[-1, ord('s'), ord('q')]):
            result = self.run_capture(camera, once=False)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1]['shot_count'], 1)
        self.assertIn('pointcloud', result[1]['frames'][0]['files'])


if __name__ == '__main__':
    unittest.main()
