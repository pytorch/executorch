# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs the Helion lowering tests for kernels on Helion's CuTe backend."""

import unittest

import executorch.backends.cuda.tests.test_lower_helion as helion_tests
import torch

try:
    from helion._compiler.cute.cutedsl_compat import check_cute_backend_requirements
except ImportError as error:
    raise unittest.SkipTest(
        f"Helion's CuTe backend is not installed: {error}"
    ) from error


def _cute_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() < (10, 0):
        return "Helion's CuTe backend targets Blackwell (SM100) GPUs"
    try:
        check_cute_backend_requirements()
    except Exception as error:  # noqa: B902 CuteBackendUnavailable is not public.
        return str(error)
    return None


_UNAVAILABLE_REASON = _cute_unavailable_reason()


@unittest.skipIf(_UNAVAILABLE_REASON is not None, _UNAVAILABLE_REASON or "")
class LowerHelionCuteTest(helion_tests.LowerHelionTest):
    BACKEND = "cute"
    LAUNCH_MARKER = "helion_cute"

    @unittest.skip("Helion's CuTe attention needs its own flash-attention config")
    def test_attention_export_and_cpp_runtime(self) -> None:
        pass

    def test_mixed_triton_and_cute_export_and_cpp_runtime(self) -> None:
        helion = helion_tests.helion
        triton_kernel = helion.kernel(
            backend="triton",
            config=helion.Config(block_sizes=[256]),
            static_shapes=True,
        )(helion_tests._add)
        cute_kernel = self._kernel(
            helion_tests._add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                first = triton_kernel(x[0], x[1])
                return cute_kernel(first, x[1])

        self._export_and_run(Model(), x, x[0] + 2 * x[1])
