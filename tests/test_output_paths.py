"""输出分配回归：重复任务、并发创建、符号链接和只读预览。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oridynamic.output_paths import output_directory


class OutputPathTests(unittest.TestCase):
    def test_first_run_and_repeated_run_preserve_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / 'runs' / 'five_cases'
            self.assertEqual(output_directory(requested), requested)
            self.assertFalse(requested.parent.exists())
            self.assertEqual(output_directory(requested, create=True), requested)
            (requested / 'result').write_bytes(b'previous result')
            preview = output_directory(requested)
            self.assertNotEqual(preview, requested)
            self.assertFalse(preview.exists())
            actual = output_directory(requested, create=True)
            self.assertTrue(actual.is_dir())
            self.assertTrue(actual.name.startswith('five_cases_'))
            self.assertEqual(actual.parent, requested.parent)
            self.assertEqual((requested / 'result').read_bytes(), b'previous result')

    def test_concurrent_creators_get_separate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / 'runs' / 'same_output'
            def allocate(index):
                out = output_directory(requested, create=True)
                (out / 'owner').write_text(str(index))
                return out, index
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(allocate, range(24)))
            self.assertEqual(len({p for p, _ in results}), 24)
            self.assertEqual(sum(p == requested for p, _ in results), 1)
            for path, index in results:
                self.assertEqual((path / 'owner').read_text(), str(index))

    def test_symlink_and_file_are_not_followed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / 'destination'; destination.mkdir()
            link = root / 'link'; link.symlink_to(destination, target_is_directory=True)
            broken = root / 'broken'; broken.symlink_to(root / 'missing')
            regular = root / 'file'; regular.write_text('keep')
            for requested in (link, broken, regular):
                out = output_directory(requested, create=True)
                self.assertNotEqual(out, requested)
                self.assertTrue(out.is_dir())
            self.assertEqual(list(destination.iterdir()), [])
            self.assertTrue(broken.is_symlink())
            self.assertEqual(regular.read_text(), 'keep')

    def test_race_after_preview_and_suffix_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / 'run'
            self.assertEqual(output_directory(requested), requested)
            requested.mkdir()  # 模拟另一个进程抢先创建。
            collision = Path(tmp) / 'run_busy'; collision.mkdir()
            available = Path(tmp) / 'run_free'
            with patch('oridynamic.output_paths.timestamped_output', side_effect=[collision, available]):
                self.assertEqual(output_directory(requested, create=True), available)
            self.assertEqual(list(requested.iterdir()), [])
            self.assertEqual(list(collision.iterdir()), [])

    def test_invalid_parent_is_not_retried_as_output_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'file'; parent.write_text('keep')
            with patch('oridynamic.output_paths.timestamped_output') as rename, self.assertRaises(OSError):
                output_directory(parent / 'run', create=True)
            rename.assert_not_called()


if __name__ == '__main__':
    unittest.main()
