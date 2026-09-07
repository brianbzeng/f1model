import json
import unittest
import pandas as pd
from refresh_forecast import ROOT, aggregate, generate


class ForecastTests(unittest.TestCase):
    def test_lapped_finish_is_not_dnf(self):
        frame = pd.DataFrame([{"constructor": "Test", "points": 0, "grid_position": 0, "finish_position": 12, "status": status} for status in ["Finished", "+1 Lap", "+2 Laps", "Engine"]])
        self.assertEqual(aggregate(frame).loc["Test", "dnf_rate"], .25)

    def test_snapshot_is_reproducible_and_probabilities_reconcile(self):
        inputs = json.loads((ROOT / "data/inputs-2026.json").read_text())
        result = generate(inputs, 2026, "2026-09-07")
        self.assertEqual(result, generate(inputs, 2026, "2026-09-07"))
        self.assertEqual(result["completedRaces"], 13)
        self.assertEqual(result["totalRaces"], 23)
        self.assertEqual(result["completedSprints"], 5)
        self.assertEqual(len(result["teams"]), 11)
        for field, total in [("title", 100), ("top3", 300), ("top5", 500)]:
            self.assertAlmostEqual(sum(row[field] for row in result["teams"]), total, places=1)
        for row in result["teams"]:
            self.assertTrue(0 <= row["title"] <= row["top3"] <= row["top5"] <= 100)
            self.assertTrue(1 <= row["meanRank"] <= 11)
        for fold in result["validation"]["folds"]:
            self.assertLess(fold["trainingThrough"], fold["season"])
        with self.assertRaises(ValueError):
            generate(inputs, 2026, "2026-07-10")


if __name__ == "__main__":
    unittest.main()
