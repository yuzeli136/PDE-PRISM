import unittest

import numpy as np
import pandas as pd

from pde_prism.ensemble import add_target_profile, aggregate
from pde_prism.filters import apply_property_filters, parse_condition, parse_task_cutoff
from pde_prism.preprocessing import process_smiles


class CoreTests(unittest.TestCase):
    def test_ensemble_mean_and_sample_sd(self):
        raw = np.zeros((10, 2, 11), dtype=float)
        raw[:, 0, 6] = np.arange(10) / 10
        result = aggregate(raw, save_individual=True)
        self.assertAlmostEqual(result.loc[0, "PDE7_mean"], 0.45)
        self.assertAlmostEqual(result.loc[0, "PDE7_sd"], np.std(np.arange(10) / 10, ddof=1))
        self.assertIn("PDE7_model_9", result)

    def test_safe_condition_parser(self):
        condition = parse_condition("PDE7>=0.85")
        frame = pd.DataFrame({"PDE7_mean": [0.84, 0.85, 0.90]})
        self.assertEqual(condition.apply(frame).tolist(), [False, True, True])
        with self.assertRaises(ValueError):
            parse_condition("__import__('os').system('x')")
        with self.assertRaises(ValueError):
            parse_condition("PDE12>=0.8")

    def test_include_exclude_parser(self):
        self.assertEqual(parse_task_cutoff("PDE4:0.8", "include").operator, ">=")
        self.assertEqual(parse_task_cutoff("PDE5:0.3", "exclude").operator, "<=")

    def test_prediction_margin(self):
        row = {f"PDE{i}_mean": 0.1 for i in range(1, 12)}
        row["PDE7_mean"] = 0.9
        row["PDE9_mean"] = 0.3
        row["PDE7_sd"] = 0.04
        result = add_target_profile(pd.DataFrame([row]), "PDE7")
        self.assertEqual(result.loc[0, "highest_offtarget_task"], "PDE9")
        self.assertAlmostEqual(result.loc[0, "prediction_margin"], 0.6)

    def test_invalid_smiles_does_not_raise(self):
        item = process_smiles("not-a-smiles")
        self.assertFalse(item.valid)
        self.assertEqual(item.status, "invalid_smiles")

    def test_property_filter_combination(self):
        frame = pd.DataFrame({"MW": [149, 300, 501], "QED": [0.5, 0.4, 0.8]})
        mask = apply_property_filters(frame, {"mw_min": 150, "mw_max": 500, "qed_min": 0.3})
        self.assertEqual(mask.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()

