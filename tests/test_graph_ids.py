import unittest
import numpy as np
from econet.graph_ids import shared_object_ids


class GraphIdentityTests(unittest.TestCase):
    def test_cross_image_pairs_share_identity(self):
        labels = shared_object_ids([[[0, 4], [1, 5]], [[2, 6], [3, 7]]], 8)
        np.testing.assert_array_equal(labels, [0, 0, 2, 2, 0, 0, 2, 2])

    def test_overlapping_graphs_merge(self):
        labels = shared_object_ids([[0, 1], [1, 4], [4, 6]], 8)
        self.assertEqual(len(set(labels[[0, 1, 4, 6]])), 1)
        self.assertEqual(labels[7], -1)

    def test_padding_is_ignored(self):
        np.testing.assert_array_equal(shared_object_ids([[-1, 1, 3, 99]], 5), [-1, 1, -1, 1, -1])

    def test_no_discovered_objects(self):
        np.testing.assert_array_equal(shared_object_ids([], 4), [-1, -1, -1, -1])
