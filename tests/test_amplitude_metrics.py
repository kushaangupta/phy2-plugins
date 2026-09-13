import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "plugins"))

from amplitude_metrics import compute_amplitude_cutoff, estimate_amplitude_cutoff


class AmplitudeCutoffTests(unittest.TestCase):
    def test_complete_symmetric_distribution_has_small_cutoff(self):
        rng = np.random.default_rng(1234)
        estimate = estimate_amplitude_cutoff(rng.normal(10.0, 1.0, 100_000))

        self.assertTrue(estimate.valid)
        self.assertLess(estimate.fraction_missing, 0.03)

    def test_truncation_increases_cutoff(self):
        rng = np.random.default_rng(1234)
        complete = rng.normal(10.0, 1.0, 200_000)
        truncated = complete[complete >= 9.5]

        full_cutoff = compute_amplitude_cutoff(complete)
        truncated_cutoff = compute_amplitude_cutoff(truncated)

        self.assertGreater(truncated_cutoff, full_cutoff + 0.1)
        self.assertLessEqual(truncated_cutoff, 0.5)

    def test_nonfinite_values_are_ignored(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=10_000)
        reference = compute_amplitude_cutoff(values)
        contaminated = compute_amplitude_cutoff(
            np.concatenate((values, [np.nan, np.inf, -np.inf]))
        )

        self.assertAlmostEqual(reference, contaminated)

    def test_small_and_constant_samples_are_unknown_not_perfect(self):
        self.assertTrue(np.isnan(compute_amplitude_cutoff(np.arange(20.0))))
        self.assertTrue(np.isnan(compute_amplitude_cutoff(np.ones(100))))

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            estimate_amplitude_cutoff(np.arange(100.0), num_bins=5)
        with self.assertRaises(ValueError):
            estimate_amplitude_cutoff(np.arange(100.0), smoothing_sigma=-1)


if __name__ == "__main__":
    unittest.main()
