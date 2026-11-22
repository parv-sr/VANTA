"""Ensure numeric policies are training-fitted, finite, and shape-safe."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "experiments" / "rn_interlude" / "robust_numeric.py"
spec = importlib.util.spec_from_file_location("robust_numeric_under_test", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestRobustNumeric(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[-1., 0.], [0., 1.], [1., 2.], [2., 3.]])
        self.test = np.array([[1e5, 1e5]])

    def test_train_only_winsor_thresholds(self):
        policy = mod.fit_policy(self.train, "winsor_01_99")
        self.assertLess(policy["upper"][0], 3)
        self.assertLess(mod.apply_policy(self.test, policy)[0, 0], 3)
        self.assertEqual(policy, mod.fit_policy(self.train, "winsor_01_99"))

    def test_standard_reference_unchanged(self):
        policy = mod.fit_policy(self.train, "standard")
        np.testing.assert_array_equal(mod.apply_policy(self.train, policy), self.train.astype(np.float32))

    def test_z_clip_bounds(self):
        out = mod.apply_policy(self.test, mod.fit_policy(self.train, "clip_z5"))
        self.assertLessEqual(np.abs(out).max(), 5)

    def test_asinh_signed_and_finite(self):
        x = np.array([[-1000., 0.], [1000., 0.]])
        out = mod.apply_policy(x, mod.fit_policy(self.train, "asinh"))
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(out[0, 0], -out[1, 0], places=5)

    def test_reject_feature_mismatch(self):
        with self.assertRaises(ValueError):
            mod.apply_policy(np.ones((2, 3)), mod.fit_policy(self.train, "clip_z5"))


if __name__ == "__main__":
    unittest.main()
