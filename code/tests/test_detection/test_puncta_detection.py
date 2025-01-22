"""
Tests the traditional detection
"""

import unittest
from aind_smartspim_cell_proposals.traditional_detection import puncta_detection
import numpy as np

class PunctaDetection(unittest.TestCase):
    """Class for testing the puncta detection"""

    def test_prune_blobs(self):
        """
        Tests prunning blobs
        """
    
        blobs_array = np.array([
            [10, 10, 10],
            [12, 15, 156],
            [6, 16, 8],
            [9, 13, 10],
            [145, 250, 356],
        ])

        expected_result = np.array(
            [10, 10, 10],
            [12, 15, 156],
            [145, 250, 356],
        )

        result = puncta_detection.prune_blobs(
            blobs_array=blobs_array,
            distance=10,
        )

        print(result)
        self.assertEqual(expected_result, result)