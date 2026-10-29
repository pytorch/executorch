# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from executorch.backends.cpu.workflows.export import (
    capture,
    image_size_for,
    main,
    synthetic_image,
)


class TorchvisionExportTest(unittest.TestCase):
    def test_pretrained_vit_h_14_input_size(self) -> None:
        self.assertEqual(518, image_size_for("vit_h_14", None))

    def test_random_weights_keep_model_defaults(self) -> None:
        self.assertEqual(224, image_size_for("vit_h_14", None, random_weights=True))
        self.assertEqual(299, image_size_for("inception_v3", None, random_weights=True))

    def test_explicit_image_size_overrides_weights(self) -> None:
        self.assertEqual(384, image_size_for("vit_h_14", 384))
        self.assertEqual(384, image_size_for("vit_h_14", 384, random_weights=True))

    def test_random_weight_vit_capture_respects_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                model="vit_b_16",
                image_size=384,
                random_weights=True,
                pt2=None,
                output=Path(directory),
            )

            exported, image = capture(args)

            self.assertEqual((1, 3, 384, 384), tuple(image.shape))
            with torch.no_grad():
                self.assertEqual((1, 1000), tuple(exported.module()(image).shape))

    def test_pretrained_vit_rejects_incompatible_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                model="vit_b_16",
                image_size=384,
                random_weights=False,
                pt2=None,
                output=Path(directory),
            )

            with self.assertRaisesRegex(ValueError, "image_size.*224.*384"):
                capture(args)

    def test_reused_capture_preserves_input_shape(self) -> None:
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 6 * 10, 4))
        image = synthetic_image((1, 3, 6, 10))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt2"
            torch.export.save(torch.export.export(model.eval(), (image,)), source)
            output = root / "output"
            output.mkdir()
            args = argparse.Namespace(model="vit_h_14", pt2=source, output=output)

            exported, actual_image = capture(args)

            torch.testing.assert_close(image, actual_image)
            torch.testing.assert_close(model(image), exported.module()(actual_image))
            self.assertEqual(image.numel() * 4, (output / "input.bin").stat().st_size)
            self.assertEqual(4 * 4, (output / "eager.bin").stat().st_size)
            self.assertEqual(
                source.read_bytes(), (output / "vit_h_14.pt2").read_bytes()
            )

    def test_reused_capture_rejects_image_size(self) -> None:
        argv = [
            "export",
            "--pt2",
            "capture.pt2",
            "--image-size",
            "8",
            "--output",
            "out",
            "--source-revision",
            "test",
        ]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
            main()

    def test_manifest_lists_nodes_from_every_cpu_delegate(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 4, 3),
        ).eval()
        captured = torch.export.export(model, (synthetic_image((1, 3, 8, 8)),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.export.save(captured, path / "capture.pt2")
            argv = [
                "export",
                "--pt2",
                str(path / "capture.pt2"),
                "--output",
                directory,
                "--source-revision",
                "test",
            ]
            with mock.patch("sys.argv", argv):
                main()
            manifest = json.loads((path / "export.json").read_text())

        self.assertEqual("capture", manifest["model"])
        self.assertEqual({"CpuBackend": 2}, manifest["delegates"])
        self.assertEqual(
            ["aten_convolution_default", "aten_convolution_default_1"],
            manifest["semantic_nodes"],
        )
