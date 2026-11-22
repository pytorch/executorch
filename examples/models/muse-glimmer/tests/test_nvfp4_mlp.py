# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 MLP adaptation keeps weights and the CPU/decode path intact."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# The CUDA NVFP4 transformation needs Triton; CPU-only and macOS jobs skip.
pytest.importorskip("triton")

from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor  # noqa: E402
from executorch.examples.models.muse_glimmer.model.model import (  # noqa: E402
    MuseGlimmerMLP,
)
from executorch.examples.models.muse_glimmer.source_transformations.nvfp4 import (  # noqa: E402
    CudaNVFP4MLP,
    install_nvfp4_gate_up_epilogues,
)


def make_mlp(nvfp4):
    config = SimpleNamespace(dim=192, ffn_dim=35, norm_eps=1e-6, fuse_gate_up=True)
    module = MuseGlimmerMLP(config).bfloat16().eval()
    if nvfp4:
        q = torch.randint(0, 256, (70, 96), dtype=torch.uint8)
        s = torch.full((70, 12), 0.0625).to(torch.float8_e4m3fn).view(torch.uint8)
        weight = CudaNVFP4Tensor(q, s, torch.tensor([0.25]))
        module.gate_up_proj.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


def test_adapter_preserves_parameters_and_cpu_fallback():
    original = make_mlp(True)
    other = make_mlp(False)
    model = SimpleNamespace(
        layers=[SimpleNamespace(mlp=original), SimpleNamespace(mlp=other)]
    )
    inputs = [torch.randn(1, m, 192, dtype=torch.bfloat16) for m in (1, 5)]
    before = [original(x) for x in inputs]
    state_keys = set(original.state_dict())
    assert install_nvfp4_gate_up_epilogues(model) == 1
    adapted = model.layers[0].mlp
    assert isinstance(adapted, CudaNVFP4MLP) and not adapted.training
    assert model.layers[1].mlp is other
    assert adapted.norm is original.norm and adapted.down_proj is original.down_proj
    assert adapted.gate_up_proj.weight is original.gate_up_proj.weight
    assert set(adapted.state_dict()) == state_keys
    for x, expected in zip(inputs, before):
        torch.testing.assert_close(adapted(x), expected, rtol=0, atol=0)


def test_adapter_rejects_unknown_activation_mode(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a8x")
    module = CudaNVFP4MLP(make_mlp(True))
    with pytest.raises(ValueError, match="ET_CUDA_NVFP4_ACT"):
        module(torch.randn(1, 5, 192, dtype=torch.bfloat16))


def test_gate_up_export_preserves_parameter_fqns(monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_gated_gemm

    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a4")
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: True)
    monkeypatch.setattr(nvfp4_gated_gemm, "_has_native_fp4_conversion", lambda: True)
    module = CudaNVFP4MLP(make_mlp(True)).eval()
    exported = torch.export.export(
        module,
        (torch.randn(1, 17, 192, dtype=torch.bfloat16),),
        dynamic_shapes=({1: torch.export.Dim("tokens", min=5, max=64)},),
        strict=True,
    ).run_decompositions()
    inputs = exported.graph_signature.input_specs
    targets = {item.target for item in inputs if item.kind.name == "PARAMETER"}
    assert {
        "gate_up_proj.parametrizations.weight.original0",
        "gate_up_proj.parametrizations.weight.original1",
        "gate_up_proj.parametrizations.weight.original2",
    } <= targets
    assert not any(item.kind.name == "CONSTANT_TENSOR" for item in inputs)
    assert any(
        "nvfp4_w4a4_gate_up" in str(node.target) for node in exported.graph.nodes
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="SM120 required",
)
def test_adapter_serialized_dynamic_m(monkeypatch):
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.runtime import Runtime

    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a4")
    torch.manual_seed(42)
    module = CudaNVFP4MLP(make_mlp(True)).eval()
    torch.manual_seed(42)
    reference = CudaNVFP4MLP(make_mlp(True)).eval().cuda()
    x = torch.randn(1, 17, 192, dtype=torch.bfloat16)
    exported = torch.export.export(
        module,
        (x,),
        dynamic_shapes=({1: torch.export.Dim("tokens", min=1, max=64)},),
        strict=True,
    )
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    from executorch.backends.cuda.cuda_weight_collector import decode_cuda_aoti_metadata
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(bytes(lowered.buffer))
    metadata = decode_cuda_aoti_metadata(
        program.program.backend_delegate_data[
            program.program.execution_plan[0].delegates[0].processed.index
        ].data
    )
    assert not any(
        entry.fqn.startswith("_tensor_constant") for entry in metadata.entries
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "mlp.pte"
        with path.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        for m in (1, 5, 17, 37, 64):
            value = torch.randn(1, m, 192, dtype=x.dtype)
            expected = reference(value.cuda()).cpu()
            actual = method.execute([value])[0].cpu()
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
