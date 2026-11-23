"""Regression checks for batch planning and successful-job artifacts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_phd08_batch import missing_outputs


class BatchIntegrityTests(unittest.TestCase):
    def test_missing_test_result_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in ("resolved_config.json", "metrics.jsonl", "last.pt"):
                (root / filename).write_text("{}", encoding="utf-8")
            attempt = {"output": str(root), "command": ["python", "train.py", "--run-test"]}
            self.assertEqual(missing_outputs({"kind": "training"}, attempt), [str(root / "test_metrics.json")])
            (root / "test_metrics.json").write_text("{}", encoding="utf-8")
            self.assertEqual(missing_outputs({"kind": "training"}, attempt), [])

    def test_baseline_comparison_and_final_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "batch"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "run_experiment_batch.py"),
                "--series", "o3", "--phase", "final", "--data-root", str(root),
                "--models-dir", str(root), "--seeds", "1", "--output-dir", str(output), "--dry-run",
            ], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            state = json.loads((output / "batch_state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["jobs"]), 8)
            for job in state["jobs"]:
                command = job["command"]
                config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
                self.assertEqual(config["train_class_key"], "final_train_classes")
                self.assertNotIn("validation", config)
                self.assertIn("--run-test", command)
                if "catml_apm_" in job["id"]:
                    self.assertEqual(config["sampler"]["apm_gamma"], 1.0)


if __name__ == "__main__":
    unittest.main()
