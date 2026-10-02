# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load_updater():
    path = ROOT / ".github" / "scripts" / "update_pytorch_pin.py"
    spec = importlib.util.spec_from_file_location("update_pytorch_pin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, body):
        self.body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


class TestUpdatePytorchPin(unittest.TestCase):
    def setUp(self):
        self.updater = load_updater()

    def test_full_wheel_version_selects_source_date(self):
        self.assertEqual(
            self.updater.parse_nightly_version("2.15.0.dev20260913"),
            "2026-09-13",
        )
        with self.assertRaises(ValueError):
            self.updater.parse_nightly_version("dev20260913")

    def test_selects_latest_complete_package_date(self):
        package_versions = {
            "torch": "2.15.0",
            "torchvision": "0.30.0",
            "torchaudio": "2.11.0",
        }

        def index(request):
            package = request.full_url.rstrip("/").rsplit("/", 1)[-1]
            channel = request.full_url.rstrip("/").rsplit("/", 2)[-2]
            platforms = (
                self.updater.CPU_WHEEL_PLATFORMS
                if channel == "cpu"
                else self.updater.CUDA_WHEEL_PLATFORMS
            )
            versions = [f"{package_versions[package]}.dev20260912"]
            if not (package == "torch" and channel == "cu134"):
                versions.append(f"{package_versions[package]}.dev20260913")
            links = "".join(
                f'<a href="{package}-{version}+{channel}-cp310-cp310-{platform}.whl">wheel</a>'
                for version in versions
                for platform in platforms
            )
            # A newer publication that omitted the oldest supported Python must
            # not displace the last complete wheel set.
            links += "".join(
                f'<a href="{package}-{package_versions[package]}.dev20260914+'
                f'{channel}-cp311-cp311-{platform}.whl">wheel</a>'
                for platform in platforms
            )
            return Response(links)

        with patch.object(self.updater.urllib.request, "urlopen", side_effect=index):
            selected = self.updater.get_pytorch_nightly_versions("20260914")
        self.assertEqual(
            selected,
            {
                package: f"{version}.dev20260912"
                for package, version in package_versions.items()
            },
        )

    def test_updates_the_single_package_configuration(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                Path("torch_pin.py").write_text((ROOT / "torch_pin.py").read_text())
                self.updater.update_pytorch_package_pins(
                    {
                        "torch": "9.0.0.dev20990101",
                        "torchvision": "8.0.0.dev20990101",
                        "torchaudio": "7.0.0.dev20990101",
                    }
                )
                config = self.updater.runpy.run_path("torch_pin.py")
            finally:
                os.chdir(original_directory)
        self.assertEqual(config["PYTORCH_VERSION"], "9.0.0.dev20990101")
        self.assertEqual(config["TORCHVISION_VERSION"], "8.0.0.dev20990101")
        self.assertEqual(config["TORCHAUDIO_VERSION"], "7.0.0.dev20990101")


if __name__ == "__main__":
    unittest.main()
