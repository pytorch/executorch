# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch
from executorch.backends.cpu.workflows.export import main
from executorch.backends.native.serialization import deserialize_graph


class WorkflowExportTest(unittest.TestCase):
    def test_native_export_uses_native_preprocessing(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1),
            torch.nn.GELU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(16, 2),
        ).eval()
        captured = torch.export.export(model, (torch.ones(1, 3, 2, 2),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.export.save(captured, path / "capture.pt2")
            argv = [
                "export",
                "--runtime",
                "native",
                "--pt2",
                str(path / "capture.pt2"),
                "--output",
                str(path),
                "--source-revision",
                "test",
            ]
            with mock.patch("sys.argv", argv):
                main()
            with zipfile.ZipFile(path / "model.ptn") as package:
                graph = deserialize_graph(package.read("program.ptg"))
                self.assertIn("program.safetensors", package.namelist())
            targets = {node.target for node in graph.nodes}
            self.assertIn("torch.ops.aten.gelu_.default", targets)
            self.assertIn("torch.ops.aten.view.default", targets)
            manifest = json.loads((path / "export.json").read_text())
            self.assertEqual("NativeBackend", manifest["preprocess"])
            self.assertEqual("native_default", manifest["recipe"]["recipe"])
            self.assertEqual([1, 3, 2, 2], manifest["input"]["shape"])
            self.assertIsNone(manifest["cpu_delegate_version"])
            self.assertFalse((path / "model.pte").exists())

    def test_native_export_rejects_cpu_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "export",
                "--runtime",
                "native",
                "--backend",
                "xnnpack",
                "--output",
                directory,
                "--source-revision",
                "test",
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(2, error.exception.code)
