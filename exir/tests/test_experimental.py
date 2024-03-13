# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional

import executorch.exir as exir

import torch
from executorch.exir import CaptureConfig
from executorch.exir.error import ExportError
from executorch.exir.experimental import add_assertions
from executorch.exir.experimental.export_pt2 import (
    ExportSession,
    Guard,
    GuardResolution,
    GuardType,
    Trace,
    trace,
)


class TestExperimental(unittest.TestCase):
    def test_assertion_inserts(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.sin(x)
            return torch.add(x, y)

        x = (torch.randn(100),)
        edge_gm = (
            exir.capture(f, x, CaptureConfig()).to_edge().exported_program.graph_module
        )
        validation_f = add_assertions(edge_gm)

        # This should run successfully since the inputs are the same size
        validation_f(torch.randn(100))

        # A shape assertion within the model should fail
        with self.assertRaises(AssertionError):
            validation_f(torch.randn(2))

        # A type assertion within the model should fail
        with self.assertRaises(AssertionError):
            validation_f(torch.randn(100, dtype=torch.float64))

    def _test_trace_export(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.cos()

        traced_object = trace(f, (torch.ones(6),))
        self.assertTrue(isinstance(traced_object, Trace))

        # user will create this
        export_session = ExportSession(traced_object)
        with self.assertRaisesRegex(ExportError, "There are outstanding guards"):
            export_session.export()

        def test_rule(guard: Guard) -> Optional[GuardResolution]:
            if guard.guard_type == GuardType.TENSOR_MATCH:
                return GuardResolution.IGNORE
            return None

        export_session.add_guard_rule(test_rule)

        graph_module = export_session.export()
        assert graph_module is not None
        self.assertTrue(torch.equal(f(torch.ones(6)), graph_module(torch.ones(6))))

        def test_rule_strict(guard: Guard) -> Optional[GuardResolution]:
            if guard.guard_type == GuardType.TENSOR_MATCH:
                return GuardResolution.ERROR_AT_EXPORT
            return None

        export_session.add_guard_rule(test_rule_strict)
        self.assertEqual(len(export_session.guard_rules), 3)
        with self.assertRaisesRegex(ExportError, "There are outstanding guards"):
            export_session.export()

        export_session.add_guard_rule(test_rule)
        self.assertEqual(len(export_session.guard_rules), 4)

        graph_module = export_session.export()
        assert graph_module is not None
        self.assertTrue(torch.equal(f(torch.ones(6)), graph_module(torch.ones(6))))
