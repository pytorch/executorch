# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def load_updater():
    path = ROOT / ".github/scripts/update_pytorch_pin.py"
    spec = importlib.util.spec_from_file_location("update_pytorch_pin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestUpdatePytorchPin(unittest.TestCase):
    def test_torchao_pin_covers_supported_nightly_channels(self):
        updater = load_updater()
        older = "0.19.0.dev20261031"
        newer = "0.19.0.dev20261101"
        versions = {
            "cpu": {older, newer},
            "cu130": {older},
            "cu132": {older, newer},
            "cu134": {older, newer},
        }

        with patch.object(
            updater, "get_torchao_versions", side_effect=versions.__getitem__
        ):
            self.assertEqual(
                updater.get_latest_torchao_nightly("20261101"),
                older,
            )

    def test_torchao_commit_accepts_a_partially_failed_wheel_run(self):
        updater = load_updater()
        run = {"head_sha": "b87db29e", "conclusion": "failure"}

        with patch.object(
            updater, "get_json", return_value={"workflow_runs": [run]}
        ) as get_json:
            self.assertEqual(
                updater.get_torchao_commit_hash("0.19.0.dev20261101"),
                "b87db29e",
            )

        self.assertNotIn("status=success", get_json.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
