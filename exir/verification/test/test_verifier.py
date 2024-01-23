# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import contextmanager

from executorch.exir.dialects._ops import ops
from torch._export.verifier import SpecViolationError
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401

from ..verifier import EXIREdgeDialectVerifier


class TestEdgeDialectVerifier(unittest.TestCase):
    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException("{} raised".format(exc_type.__name__))

    def test_edge_verifier_check_valid_op_succeed_given_custom_op(self) -> None:
        edge_op = ops.edge.quantized_decomposed.quantize_per_tensor.default
        verifier = EXIREdgeDialectVerifier(check_edge_ops=True)
        with self.assertNotRaises(SpecViolationError):
            verifier.check_valid_edge_op(edge_op)
            verifier.check_valid_op(edge_op)
