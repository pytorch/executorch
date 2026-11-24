# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for the routing tests: the graphs the CUDA backend receives, and
compile specs for tests that export calls to check numerics or the static
routing rather than the measurement (see triton/measured_routing.py), so they
do not depend on timings or pay for measuring."""

import contextlib
from unittest import mock

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec

LOWERING_APIS = ("to_edge_transform_and_lower", "to_edge_to_backend")


def static_routing_compile_specs(method: str = "forward") -> list[CompileSpec]:
    return [
        CudaBackend.generate_method_name_compile_spec(method),
        CompileSpec("measured_routing", b"OFF"),
    ]


def backend_graphs(model, *inputs, dynamic_shapes=None, device="cuda"):
    """The graph the CUDA backend gets from each lowering API.

    The model is exported on fake CUDA tensors, or on CPU tensors whose fake
    values are then moved to CUDA, as the backend does before the pass runs.
    """
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import to_edge, to_edge_transform_and_lower
    from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    with fake_mode if device == "cuda" else contextlib.nullcontext(), torch.device(
        device
    ):
        model = model()
        args = tuple(make() for make in inputs)
        program = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
    captured = []

    def capture(cls, edge_program, compile_specs, collector):
        captured.append(edge_program.graph_module)
        return PreprocessResult(processed_bytes=b"")

    # Lowering looks the backend up by class name; patch every class of that
    # name, since the module may have been imported under another name too.
    backends = {CudaBackend}
    backends.update(
        cls for cls in BackendDetails.__subclasses__() if cls.__name__ == "CudaBackend"
    )
    with contextlib.ExitStack() as stack:
        for backend in backends:
            stack.enter_context(
                mock.patch.object(
                    backend, "_preprocess_with_weight_collector", classmethod(capture)
                )
            )
        to_edge_transform_and_lower(program, partitioner=[CudaPartitioner([])])
        to_edge(program).to_backend(CudaPartitioner([]))
    for graph_module in captured if device == "cpu" else ():
        for node in graph_module.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                with fake_mode:
                    node.meta["val"] = torch.empty_strided(
                        val.shape, val.stride(), dtype=val.dtype, device="cuda"
                    )
    return dict(zip(LOWERING_APIS, captured))
