# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import Case, draws, suite_cases
from executorch.backends.native.test.samples.record import record
from executorch.backends.native.test.samples.schemes import Scheme, schemes
from executorch.exir.native import read_samples


class RecordTest(unittest.TestCase):
    def test_suite_cases_raises_on_a_model_eager_rejects(self) -> None:
        module = ModuleType("suite")
        module.test_mismatch = lambda runner: runner.lower_and_run_model(
            torch.nn.Linear(4, 2), (torch.randn(1, 3),)
        )
        with self.assertRaises(RuntimeError) as raised:
            suite_cases(module, samples=1)
        self.assertIn(
            "while recording suite case mismatch_0", raised.exception.__notes__
        )

    def test_suite_cases_raises_when_seeds_lower_different_shapes(self) -> None:
        module = ModuleType("suite")
        sizes = iter([2, 3])
        module.test_varying = lambda runner: runner.lower_and_run_model(
            torch.nn.Identity(), (torch.randn(next(sizes)),)
        )
        with self.assertRaises(ValueError):
            suite_cases(module, samples=2)

    def test_suite_cases_raises_when_no_test_lowers_a_model(self) -> None:
        with self.assertRaises(ValueError):
            suite_cases(ModuleType("suite"), samples=1)

    def test_record_raises_on_duplicate_case_names(self) -> None:
        case = Case("same", torch.nn.Linear(2, 2), draws(1, (1, 2)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                record([case, case], Scheme(), Path(tmp) / "dup.ptn")

    def test_record_writes_no_package_when_the_scheme_fits_no_case(self) -> None:
        case = Case("relu", torch.nn.ReLU(), draws(1, (1, 32)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relu.ptn"
            path.write_bytes(b"stale")
            record([case], schemes()["int4wo_g32"], path)
            self.assertFalse(path.exists())

    def test_record_golden_is_eager_and_precise_is_float64(self) -> None:
        model = torch.nn.Linear(16, 4)
        (x,) = draws(1, (3, 16))[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.ptn"
            record([Case("linear", model, [(x,)])], Scheme(), path, dtypes=["bf16"])
            sets = read_samples(str(path))
        x, w, b = (t.detach().bfloat16() for t in (x, model.weight, model.bias))
        torch.testing.assert_close(
            sets["golden"].samples[0].calls[0].outputs[0],
            F.linear(x, w, b),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sets["precise"].samples[0].calls[0].outputs[0],
            F.linear(x.double(), w.double(), b.double()).float(),
            rtol=0,
            atol=0,
        )

    def test_record_adds_the_8da4w_golden_to_weight_only_packages(self) -> None:
        case = Case("linear", torch.nn.Linear(64, 8), draws(2, (3, 64)))
        sets = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("float", "int4wo_g32", "8da4w_g32"):
                path = Path(tmp) / f"{name}.ptn"
                record([case], schemes()[name], path, dtypes=["f32"])
                sets[name] = read_samples(str(path))
        self.assertNotIn("golden-int8_act", sets["float"])
        self.assertNotIn("golden-int8_act", sets["8da4w_g32"])
        for got, want in zip(
            sets["int4wo_g32"]["golden-int8_act"].samples,
            sets["8da4w_g32"]["golden"].samples,
            strict=True,
        ):
            torch.testing.assert_close(
                got.calls[0].outputs, want.calls[0].outputs, rtol=0, atol=0
            )

    def test_record_keeps_integer_outputs_precise(self) -> None:
        class Argmax(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.argmax(-1)

        case = Case("argmax", Argmax(), draws(1, (2, 8)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "argmax.ptn"
            record([case], Scheme(), path, dtypes=["f32"])
            precise = read_samples(str(path))["precise"].samples
        self.assertEqual(precise[0].calls[0].outputs[0].dtype, torch.int64)
