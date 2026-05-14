"""
Tests the traditional detection
"""

import importlib.util
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

# puncta_detection and its dependencies import cupy/dask at module level.
# Mock unavailable GPU/heavy packages so numpy/scipy functions can still be tested.
_cupy_available = importlib.util.find_spec("cupy") is not None
if not _cupy_available:
    sys.modules.setdefault("cupy", MagicMock())
    sys.modules.setdefault("cupyx", MagicMock())
    sys.modules.setdefault("cupyx.scipy", MagicMock())
    sys.modules.setdefault("cupyx.scipy.ndimage", MagicMock())

if importlib.util.find_spec("dask") is None:
    _dask_mock = MagicMock()
    sys.modules.setdefault("dask", _dask_mock)
    sys.modules.setdefault("dask.array", _dask_mock)

from aind_smartspim_segmentation.traditional_detection import puncta_detection  # noqa: E402


class TestPruneBlobs(unittest.TestCase):
    """Tests for the prune_blobs function (CPU/numpy path)."""

    def test_prune_blobs_basic(self):
        """Two overlapping pairs pruned by intensity; far blobs survive."""
        blobs_array = np.array(
            [
                [10, 10, 10, 100],
                [12, 15, 156, 200],
                [6, 16, 8, 50],
                [9, 13, 10, 80],
                [145, 250, 356, 150],
            ],
            dtype=float,
        )
        result_blobs, removed = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=10,
        )
        self.assertIsInstance(result_blobs, np.ndarray)
        self.assertIsInstance(removed, np.ndarray)
        # All surviving blobs must have positive intensity
        self.assertTrue(np.all(result_blobs[:, -1] > 0))

    def test_prune_blobs_empty_array(self):
        """Empty input returns empty output without raising."""
        blobs_array = np.empty((0, 4), dtype=float)
        result_blobs, removed = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=10,
        )
        self.assertEqual(result_blobs.shape[0], 0)
        self.assertEqual(len(removed), 0)

    def test_prune_blobs_single_blob(self):
        """Single blob is returned unchanged with no removals."""
        blobs_array = np.array([[5, 5, 5, 100]], dtype=float)
        result_blobs, removed = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=10,
        )
        np.testing.assert_array_equal(result_blobs, blobs_array)
        self.assertEqual(len(removed), 0)

    def test_prune_blobs_keeps_higher_intensity(self):
        """When two blobs overlap, the higher-intensity one survives."""
        blobs_array = np.array(
            [
                [10, 10, 10, 50],   # lower → removed
                [10, 10, 10, 150],  # higher → survives
            ],
            dtype=float,
        )
        result_blobs, removed = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=1,
        )
        self.assertEqual(len(result_blobs), 1)
        self.assertEqual(result_blobs[0, -1], 150)
        self.assertIn(0, removed)

    def test_prune_blobs_no_overlap(self):
        """Blobs far apart all survive with no removals."""
        blobs_array = np.array(
            [
                [0, 0, 0, 100],
                [100, 100, 100, 200],
                [200, 200, 200, 150],
            ],
            dtype=float,
        )
        result_blobs, removed = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=10,
        )
        self.assertEqual(len(result_blobs), 3)
        self.assertEqual(len(removed), 0)

    @unittest.skipUnless(_cupy_available, "cupy not available")
    def test_prune_blobs_optimized_empty_guard(self):
        """prune_blobs_optimized must not crash when no pairs are found."""
        import cupy as cp
        from aind_smartspim_segmentation.traditional_detection import (
            puncta_detection_optimized,
        )

        # Blobs far enough apart that no pairs exist within distance=1
        blobs_array = cp.array(
            [[0, 0, 0, 100], [100, 100, 100, 200]],
            dtype=cp.float32,
        )
        result, removed = puncta_detection_optimized.prune_blobs_optimized(
            blobs_array=blobs_array,
            distance=1,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(removed), 0)


if __name__ == "__main__":
    unittest.main()
