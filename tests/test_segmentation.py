import json
from pathlib import Path
import unittest
import numpy as np
from econet.pegbis.egbis import segment
from econet.pegbis.disjoint_set import universe


class SegmentationTests(unittest.TestCase):
    def test_uploaded_kernel_regression(self):
        cases = json.loads((Path(__file__).parent/'segmentation_cases.json').read_text())
        for case in cases:
            image = np.zeros((6, 7, case['channels']), dtype=float)
            image[:3, :3] = 90
            image[3:, 3:] = 180
            actual = segment(image, 0, 50, 1, '', four_connect=case['four_connect'])
            np.testing.assert_array_equal(actual, case['labels'])

    def test_uniform_region(self):
        labels = segment(np.ones((5, 5, 3))*100, 0, 50, 1, '')
        self.assertEqual(len(np.unique(labels)), 1)

    def test_disjoint_sets(self):
        sets = universe(5)
        sets.join(sets.find(0), sets.find(1))
        sets.join(sets.find(1), sets.find(2))
        self.assertEqual(sets.find(0), sets.find(2))
        self.assertEqual(sets.num_sets(), 3)
        self.assertEqual(sets.size(sets.find(0)), 3)
