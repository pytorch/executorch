# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile specs for tests that export convolutions to check numerics or the
static routing rather than the measurement (see triton/measured_routing.py),
so they do not depend on timings or pay for measuring."""

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec


def static_routing_compile_specs(method: str = "forward") -> list[CompileSpec]:
    return [
        CudaBackend.generate_method_name_compile_spec(method),
        CompileSpec("measured_routing", b"OFF"),
    ]
