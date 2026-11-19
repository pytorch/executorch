# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    cell_step,
    CheckOffGraphKVStepWidthPass,
    LowerOffGraphKVPass,
    OFFGRAPH_KV_CELLS_FQN,
    OFFGRAPH_KV_COMPILE_SPEC,
    offgraph_kv_mask_fqn,
    OFFGRAPH_KV_READ_LEN_FQN,
    OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    parse_offgraph_kv_manifest,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.llm.cache.update_and_attend import (  # noqa: F401
    update_and_attend,
)


SCALE = 0.125


class _DraftAttention(torch.nn.Module):
    def __init__(self, transform="select"):
        super().__init__()
        self.transform = transform

    def forward(self, q, k, v, position, order):
        if self.transform == "select":
            position = position.index_select(0, order)
        elif self.transform == "short_select":
            position = position.index_select(0, order[:-1])
        elif self.transform == "wrong_dim_select":
            position = position.reshape(-1, 1).index_select(1, order[:1] * 0)
        elif self.transform == "add":
            position = position + 1
        elif self.transform == "flip":
            position = position.flip(0)
        elif self.transform == "double_select":
            position = position.index_select(0, order).index_select(0, order)
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position.reshape(-1, 1), 0, SCALE, torch.bfloat16
        )


def _args(device="cpu", q_width=4, kv_width=8):
    q = torch.ones(1, 4, q_width, 64, device=device, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, kv_width, 64, device=device, dtype=torch.bfloat16)
    v = torch.arange(kv_width, device=device, dtype=torch.bfloat16)
    v = v.reshape(1, 1, kv_width, 1).expand_as(k).contiguous()
    return (
        q,
        k,
        v,
        torch.arange(kv_width, device=device),
        torch.arange(kv_width - 1, -1, -1, device=device),
    )


def _edge(transform="select", args=None, dynamic=False):
    args = _args() if args is None else args
    shapes = None
    if dynamic:
        queries = torch.export.Dim("queries", min=2, max=16)
        writes = torch.export.Dim("writes", min=2, max=32)
        shapes = ({2: queries}, {2: writes}, {2: writes}, {0: writes}, {0: writes})
    program = torch.export.export(
        _DraftAttention(transform), args, dynamic_shapes=shapes, strict=False
    )
    return to_edge(
        program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()


def _manifest(max_cells=512):
    return json.dumps(
        {
            "version": 1,
            "layout": "cell",
            "maximum_capacity": max_cells,
            "max_cells": max_cells,
            "max_write": 16,
            "layers": [{"layer_id": 0, "policy": "flat"}],
        }
    ).encode()


@pytest.mark.parametrize("dynamic", [False, True])
def test_cell_checker_accepts_full_width_position_select(dynamic):
    edge = _edge(dynamic=dynamic)
    assert CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(edge) is edge


@pytest.mark.parametrize("transform", ["select", "add", "flip", "double_select"])
def test_ordinary_checker_remains_strict(transform):
    with pytest.raises(ValueError, match="position comes from"):
        CheckOffGraphKVStepWidthPass(b"3:0")(_edge(transform))


@pytest.mark.parametrize("transform", ["add", "flip", "double_select"])
def test_cell_checker_rejects_unvalidated_transforms(transform):
    with pytest.raises(ValueError, match="position comes from"):
        CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(_edge(transform))


@pytest.mark.parametrize("transform", ["short_select", "wrong_dim_select"])
def test_cell_checker_rejects_non_full_token_axis_select(transform):
    with pytest.raises(ValueError, match="preserve full width"):
        CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(_edge(transform))


@pytest.mark.parametrize(
    "spec,error",
    [(b"1:2", "position comes from"), (b"3:1", "not the step"), (b"5:0", "takes 5")],
)
def test_cell_checker_validates_delegate_input_and_dimension(spec, error):
    with pytest.raises(ValueError, match=error):
        CheckOffGraphKVStepWidthPass(spec, layout="cell")(_edge())


@pytest.mark.parametrize("which", [1, 2])
def test_cell_checker_rejects_mismatched_kv_extent(which):
    args = list(_args())
    args[which] = args[which][:, :, :-1].contiguous()
    with pytest.raises(ValueError, match="matching K/V extent"):
        CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(_edge(args=tuple(args)))


def test_ordinary_checker_accepts_direct_position():
    edge = _edge("direct", args=_args(q_width=8))
    assert CheckOffGraphKVStepWidthPass(b"3:0")(edge) is edge


def test_backend_uses_manifest_layout_for_checker_independent_of_spec_order():
    for layout in ("cell", "sequence"):
        manifest = dict(json.loads(_manifest()), layout=layout)
        specs = [
            CompileSpec(OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, b"3:0"),
            CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, json.dumps(manifest).encode()),
        ]
        for ordered in (specs, list(reversed(specs))):
            passes = CudaBackend.get_custom_passes(ordered)
            checker = next(
                p for p in passes if isinstance(p, CheckOffGraphKVStepWidthPass)
            )
            assert passes.index(checker) < next(
                i for i, p in enumerate(passes) if isinstance(p, LowerOffGraphKVPass)
            )
            if layout == "cell":
                assert checker(_edge()) is not None
            else:
                with pytest.raises(ValueError, match="position comes from"):
                    checker(_edge())


def test_cpu_lowering_uses_kv_cells_and_query_mask_rows():
    edge = _edge()
    CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(edge)
    lowered = LowerOffGraphKVPass(parse_offgraph_kv_manifest(_manifest()))(edge)
    nodes = list(lowered.graph.nodes)
    assert not any(
        n.target == exir_ops.edge.kvcache.update_and_attend.default for n in nodes
    )
    copies = [n for n in nodes if n.target == torch.ops.aten.index_copy_.default]
    assert len(copies) == 2
    assert all(tuple(n.args[2].meta["val"].shape) == (8,) for n in copies)
    (attention,) = [n for n in nodes if n.target == torch.ops.triton.sdpa.default]
    assert tuple(attention.args[3].meta["val"].shape) == (1, 1, 4, 512)
    assert attention.args[5] is False


def _runtime(device, read_len):
    args = _args(device)
    q, k, v = args[:3]
    # Owner 0: retained cell 0, three context rows, two block rows.
    # Owner 1: retained cell 1, one context row, two block rows.
    slots = torch.tensor([3, 6, 2, 5, 4, 9, 7, 8], device=device)
    cells = torch.full((16,), 511, dtype=torch.long, device=device)
    cells[:8] = slots
    k_pool = torch.full((1, 512, 2, 64), 10000.0, dtype=q.dtype, device=device)
    v_pool = torch.full_like(k_pool, -10000.0)
    k_pool[:, :2].zero_()
    v_pool[:, 0].fill_(10)
    v_pool[:, 1].fill_(20)
    mask = torch.zeros(1, 1, 16, 512, dtype=torch.bool, device=device)
    mask[0, 0, :2, [0, 3, 6, 2, 5, 4]] = True
    mask[0, 0, 2:4, [1, 9, 7, 8]] = True
    # Stale visible tail must be excluded by the device scalar, not the mask.
    mask[..., read_len:] = True
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                q[:, :, :2].float(),
                torch.cat([k_pool[:, :1].transpose(1, 2), k[:, :, :5]], 2).float(),
                torch.cat([v_pool[:, :1].transpose(1, 2), v[:, :, :5]], 2).float(),
                scale=SCALE,
                enable_gqa=True,
            ),
            F.scaled_dot_product_attention(
                q[:, :, 2:].float(),
                torch.cat([k_pool[:, 1:2].transpose(1, 2), k[:, :, 5:]], 2).float(),
                torch.cat([v_pool[:, 1:2].transpose(1, 2), v[:, :, 5:]], 2).float(),
                scale=SCALE,
                enable_gqa=True,
            ),
        ],
        2,
    )
    return (
        args,
        (k_pool, v_pool, cells, torch.tensor([read_len], device=device), mask),
        expected,
    )


def _cpu_sdpa(q, k, v, mask, dropout, causal, scale, gqa, read_len):
    extent = int(read_len.item())
    return F.scaled_dot_product_attention(
        q.float(),
        k[:, :, :extent].float(),
        v[:, :, :extent].float(),
        attn_mask=mask[..., :extent],
        scale=scale,
        enable_gqa=gqa,
    ).to(q.dtype)


def test_cpu_cell_step_writes_all_kv_and_attends_only_block_queries():
    args, runtime, expected = _runtime("cpu", 10)
    with patch.object(torch.ops.triton, "sdpa", _cpu_sdpa):
        actual = cell_step(*args[:3], *runtime, SCALE, torch.float32)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0)
    k_pool, v_pool, cells = runtime[:3]
    torch.testing.assert_close(k_pool[:, cells[:8]], args[1].transpose(1, 2))
    torch.testing.assert_close(v_pool[:, cells[:8]], args[2].transpose(1, 2))
    assert torch.all(k_pool[:, 10:] == 10000)
    assert torch.all(v_pool[:, 10:] == -10000)


@pytest.mark.parametrize("mode", ["eager", "compile", "graph"])
@pytest.mark.parametrize("read_len", [10, 300])
def test_gpu_unequal_width_owner_visibility_and_poisoned_tail(mode, read_len):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    args, runtime, expected = _runtime("cuda", read_len)
    fn = cell_step
    if mode != "eager":
        fn = torch.compile(cell_step, fullgraph=True)
    inputs = (*args[:3], *runtime, SCALE)
    actual = fn(*inputs)
    if mode == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fn(*inputs)
        # Replay must consume new K/V and runtime side-buffer data at fixed addresses.
        args[2].add_(2)
        expected[:, :, :2].add_(2 * 5 / 6)
        expected[:, :, 2:].add_(2 * 3 / 4)
        runtime[3].fill_(10)
        runtime[4][..., 10:] = True
        graph.replay()
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0)
    k_pool, v_pool, cells = runtime[:3]
    torch.testing.assert_close(k_pool[:, cells[:8]], args[1].transpose(1, 2))
    torch.testing.assert_close(v_pool[:, cells[:8]], args[2].transpose(1, 2))
    assert torch.all(k_pool[:, read_len:] == 10000)
    assert torch.all(v_pool[:, read_len:] == -10000)


def test_gpu_lowered_draft_program_matches_owner_reference():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    args, buffers, expected = _runtime("cuda", 10)
    edge = _edge(args=args)
    CheckOffGraphKVStepWidthPass(b"3:0", layout="cell")(edge)
    lowered = LowerOffGraphKVPass(parse_offgraph_kv_manifest(_manifest()))(edge)
    names = (
        "__et_offgraph_kv_layer_0_k",
        "__et_offgraph_kv_layer_0_v",
        OFFGRAPH_KV_CELLS_FQN,
        OFFGRAPH_KV_READ_LEN_FQN,
        offgraph_kv_mask_fqn(0),
    )
    for name, value in zip(names, buffers):
        lowered.constants[name] = value
    actual = lowered.module()(*args)
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0)
