# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.exir.dialects._ops import bind_pattern_to_op, ops
from executorch.exir.dialects.backend._ops import BackendOpOverload
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.library import impl, Library

lib = Library("DO_NOT_USE_TEST_ONLY", "DEF")

lib.define("foo(Tensor self) -> Tensor")


@impl(lib, "foo", "CompositeExplicitAutograd")
def foo(a):
    return a


# schema only
lib.define("bar(Tensor self) -> Tensor")


lib.define("meta_only(Tensor self) -> Tensor")


@impl(lib, "meta_only", "Meta")
def meta(a):
    return torch.empty_like(a)


class TestBackendOps(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_foo: OpOverload = torch.ops.DO_NOT_USE_TEST_ONLY.foo.default
        self.edge_foo: EdgeOpOverload = ops.edge.DO_NOT_USE_TEST_ONLY.foo.default
        self.backend_foo: BackendOpOverload = (
            ops.backend.DO_NOT_USE_TEST_ONLY.foo.default
        )

    def test_backend_ops_with_default_kernel_is_callable(self):
        self.assertTrue(callable(self.backend_foo))
        a = torch.randn(2, 3)
        self.assertTrue(self.torch_foo(a).allclose(self.backend_foo(a)))
        self.assertTrue(self.edge_foo(a).allclose(self.backend_foo(a)))

    def test_backend_ops_with_meta_kernel_passes(self):
        x = torch.randn(2, 3)
        with FakeTensorMode() as mode:
            pass
        out = FakeTensor.from_tensor(x, mode)
        with mode:
            meta_result = ops.backend.DO_NOT_USE_TEST_ONLY.meta_only.default(out)

        self.assertEqual(meta_result.size(), x.size())

    def test_backend_ops_equivalent_pattern(self):
        with self.assertRaises(AssertionError):
            ops.backend.DO_NOT_USE_TEST_ONLY.bar.default

        @bind_pattern_to_op(lib, "bar")
        def f(x: torch.Tensor):
            return x + x

        op = ops.backend.DO_NOT_USE_TEST_ONLY.bar.default

        a = torch.randn(2, 3)
        self.assertTrue(op(a).allclose(f(a)))

    def test_backend_ops_with_schema(self):
        @bind_pattern_to_op(
            lib, "DO_NOT_USE_TEST_ONLY::foo.Tensor(Tensor self) -> Tensor"
        )
        def f(x: torch.Tensor):
            return x + x

        op = ops.backend.DO_NOT_USE_TEST_ONLY.foo.Tensor

        a = torch.randn(2, 3)
        self.assertTrue(op(a).allclose(f(a)))
