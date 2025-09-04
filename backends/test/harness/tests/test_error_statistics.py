import unittest

import torch
from executorch.backends.test.harness.error_statistics import ErrorStatistics


class ErrorStatisticsTests(unittest.TestCase):
    def test_error_stats_simple(self):
        tensor1 = torch.tensor([1, 2, 3, 4])
        tensor2 = torch.tensor([2, 2, 2, 5])

        error_stats = ErrorStatistics.from_tensors(tensor1, tensor2)

        # Check actual tensor statistics
        self.assertEqual(error_stats.actual_stats.shape, torch.Size([4]))
        self.assertEqual(error_stats.actual_stats.numel, 4)
        self.assertEqual(error_stats.actual_stats.median, 2.5)
        self.assertEqual(error_stats.actual_stats.mean, 2.5)
        self.assertEqual(error_stats.actual_stats.max, 4)
        self.assertEqual(error_stats.actual_stats.min, 1)

        # Check reference tensor statistics
        self.assertEqual(error_stats.reference_stats.shape, torch.Size([4]))
        self.assertEqual(error_stats.reference_stats.numel, 4)
        self.assertEqual(error_stats.reference_stats.median, 2.0)
        self.assertEqual(error_stats.reference_stats.mean, 2.75)
        self.assertEqual(error_stats.reference_stats.max, 5)
        self.assertEqual(error_stats.reference_stats.min, 2)

        # Check error statistics
        self.assertAlmostEqual(error_stats.error_l2_norm, 1.732, places=3)
        self.assertEqual(error_stats.error_mae, 0.75)
        self.assertEqual(error_stats.error_max, 1.0)
        self.assertEqual(error_stats.error_msd, -0.25)
        self.assertAlmostEqual(error_stats.sqnr, 10.0, places=3)

    def test_error_stats_different_shapes(self):
        # Create tensors with different shapes
        tensor1 = torch.tensor([1, 2, 3, 4])
        tensor2 = torch.tensor([[2, 3], [4, 5]])

        error_stats = ErrorStatistics.from_tensors(tensor1, tensor2)

        # Check actual tensor statistics
        self.assertEqual(error_stats.actual_stats.shape, torch.Size([4]))
        self.assertEqual(error_stats.actual_stats.numel, 4)
        self.assertEqual(error_stats.actual_stats.median, 2.5)
        self.assertEqual(error_stats.actual_stats.mean, 2.5)
        self.assertEqual(error_stats.actual_stats.max, 4)
        self.assertEqual(error_stats.actual_stats.min, 1)

        # Check reference tensor statistics
        self.assertEqual(error_stats.reference_stats.shape, torch.Size([2, 2]))
        self.assertEqual(error_stats.reference_stats.numel, 4)
        self.assertEqual(error_stats.reference_stats.median, 3.5)
        self.assertEqual(error_stats.reference_stats.mean, 3.5)
        self.assertEqual(error_stats.reference_stats.max, 5)
        self.assertEqual(error_stats.reference_stats.min, 2)

        # Check that all error values are None when shapes differ
        self.assertIsNone(error_stats.error_l2_norm)
        self.assertIsNone(error_stats.error_mae)
        self.assertIsNone(error_stats.error_max)
        self.assertIsNone(error_stats.error_msd)
        self.assertIsNone(error_stats.sqnr)
