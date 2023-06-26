#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

from executorch.exir.dialects._ops import _DialectNamespace, _Ops, ops


class TestExirDialectOps(unittest.TestCase):
    def test_ops(self) -> None:
        self.assertTrue(isinstance(ops, _Ops))

    def test_backend_op_should_not_contain_aten_op(self) -> None:
        self.assertTrue(isinstance(ops.edge, _DialectNamespace))
        with self.assertRaisesRegex(
            RuntimeError, "op library does not belong to backend ops"
        ):
            ops.backend.aten
