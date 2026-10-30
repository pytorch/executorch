# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from executorch_native import Model


class BindingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.package = Path(os.environ["NATIVE_TEST_PACKAGE"]).resolve()
        self.engine = os.environ.get("NATIVE_TEST_ENGINE", "ggml")
        self.model = Model.load(self.package)
        self.session = self.model.create_session(self.engine)
        self.input = np.arange(1, 7, dtype=np.float32).reshape(1, 6)

    def test_metadata_and_session_lifetime(self) -> None:
        self.assertEqual(self.model.method_names, ["forward"])
        info = self.model.method_info()
        del self.model
        gc.collect()
        self.assertEqual(info.name, "forward")
        self.assertEqual(info.inputs[0].shape, [1, 6])
        self.assertEqual(info.outputs[0].dtype, np.float32)
        (output,) = self.session.run([self.input])
        np.testing.assert_array_equal(output, self.input * 2)

    def test_output_ownership_and_strided_input(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(1, 12)[:, ::-2]
        data.flags.writeable = False
        (first,) = self.session.run([data])
        (second,) = self.session.run([self.input])
        del self.session
        gc.collect()
        self.assertTrue(first.flags.owndata)
        np.testing.assert_array_equal(first, data * 2)
        np.testing.assert_array_equal(second, self.input * 2)

    def test_invalid_inputs_do_not_poison_session(self) -> None:
        for inputs, error, message in [
            ([], ValueError, "count"),
            ([self.input.astype(np.float64)], TypeError, "dtype"),
            ([self.input.astype(">f4")], TypeError, "dtype"),
            ([self.input.reshape(6)], ValueError, "shape"),
            ([self.input.tolist()], TypeError, "NumPy"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    self.session.run(inputs)
        with self.assertRaisesRegex(ValueError, "unknown method"):
            self.session.run([self.input], method="missing")
        (output,) = self.session.run([self.input])
        np.testing.assert_array_equal(output, self.input * 2)

    def test_concurrent_calls_are_serialized(self) -> None:
        inputs = [self.input + i for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda x: self.session.run([x])[0], inputs))
        for actual, source in zip(results, inputs):
            np.testing.assert_array_equal(actual, source * 2)

    def test_invalid_engine(self) -> None:
        with self.assertRaisesRegex(ValueError, "engine must be"):
            self.model.create_session("unknown")

    def test_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.ptn"
            with self.assertRaisesRegex(RuntimeError, "cannot open") as error:
                Model.load(missing)
            self.assertIn(str(missing), str(error.exception))

    def test_vision_example(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.bin"
            expected_path = Path(directory) / "expected.bin"
            categories_path = Path(directory) / "categories.txt"
            self.input.tofile(input_path)
            (self.input * 2).tofile(expected_path)
            categories_path.write_text("\n".join(f"label_{i}" for i in range(6)))
            for extra_args in (
                [],
                [
                    "--expected",
                    str(expected_path),
                    "--categories",
                    str(categories_path),
                ],
            ):
                with self.subTest(extra_args=extra_args):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "executorch_native.vision",
                            "--package",
                            str(self.package),
                            "--input",
                            str(input_path),
                            "--engine",
                            self.engine,
                            *extra_args,
                        ],
                        cwd=directory,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if extra_args:
                        self.assertIn("PASS: logits match", result.stdout)
                        self.assertIn("label_5", result.stdout)
                    else:
                        self.assertNotIn("PASS: logits match", result.stdout)
                        self.assertIn("class_5", result.stdout)
