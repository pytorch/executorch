# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
if ROOT.name == "oss":
    ROOT = ROOT.parent
SPEC = importlib.util.spec_from_file_location(
    "wheel_version", ROOT / "backends/native/python/wheel_version.py"
)
assert SPEC is not None and SPEC.loader is not None
VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSION)


class NativeWheelVersionTest(unittest.TestCase):
    def test_invalid_metadata_configuration(self) -> None:
        for field, settings in (
            ("dependencies", {}),
            ("version", {"input": "version.txt"}),
        ):
            with self.subTest(field=field, settings=settings):
                with self.assertRaisesRegex(
                    ValueError, "Only version metadata without settings is supported"
                ):
                    VERSION.dynamic_metadata(field, settings)

    def test_build_version_overrides_source_version(self) -> None:
        for version in ("1.6.0", "1.6.0rc2", "1.7.0.dev20261021"):
            with self.subTest(version=version), patch.dict(
                os.environ, {"BUILD_VERSION": f" {version}\n"}
            ):
                self.assertEqual(VERSION.dynamic_metadata("version", {}), version)

    def test_local_build_uses_version_file(self) -> None:
        for version in ("", " \n"):
            with self.subTest(version=version), patch.dict(
                os.environ, {"BUILD_VERSION": version}
            ):
                self.assertEqual(
                    VERSION.dynamic_metadata("version", {}),
                    (ROOT / "version.txt").read_text().strip(),
                )


if __name__ == "__main__":
    unittest.main()
