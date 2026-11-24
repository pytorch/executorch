# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 GGUF loading: companion scale pairing and CUDA fusion."""

import os
import tempfile
from enum import IntEnum
from unittest import mock

import numpy as np
import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
    _atomic_sd_from_gguf,
    _fuse_state_dict,
)
from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor

try:
    import gguf

    HAS_GGUF = True
    HAS_GGUF_NVFP4 = hasattr(gguf.GGMLQuantizationType, "NVFP4")
except ImportError:
    HAS_GGUF = HAS_GGUF_NVFP4 = False
# Only the tests that write GGUF files need gguf >= 0.19.
requires_gguf = pytest.mark.skipif(not HAS_GGUF_NVFP4, reason="needs gguf >= 0.19")


def nvfp4_raw(n, k, seed):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(
        0, 256, (n, k // 64 * 36), dtype=torch.uint8, generator=generator
    )
    raw.reshape(n, -1, 36)[..., :4] = 0x38
    return raw.numpy()


def write_gguf(path, tensors):
    writer = gguf.GGUFWriter(path, "muse_glimmer")
    for name, value in tensors:
        if value.dtype == np.uint8:
            writer.add_tensor(name, value, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        else:
            writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def scalar(value):
    return np.array([value], dtype=np.float32)


def load(tensors, static_act_scale=False):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.gguf")
        write_gguf(path, tensors)
        return _atomic_sd_from_gguf(path, torch.bfloat16, static_act_scale)


# Companions before and after their weight, and a weight without any.
TENSORS = [
    ("blk.0.ffn_down.scale", scalar(0.25)),
    ("blk.0.ffn_down.weight", nvfp4_raw(8, 128, 0)),
    ("blk.0.ffn_down.input_scale", scalar(0.002)),
    ("blk.0.ffn_up.weight", nvfp4_raw(8, 128, 1)),
    ("blk.0.ffn_up.scale", scalar(0.5)),
    ("blk.0.ffn_gate.weight", nvfp4_raw(8, 128, 2)),
    ("blk.0.ffn_norm.weight", np.ones(128, dtype=np.float32)),
]


@requires_gguf
def test_gguf_nvfp4_companions_pair_by_name():
    sd = load(TENSORS)
    assert set(sd) == {
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.norm.weight",
    }
    down = sd["layers.0.mlp.down_proj.weight"]
    up = sd["layers.0.mlp.up_proj.weight"]
    gate = sd["layers.0.mlp.gate_proj.weight"]
    assert all(isinstance(w, ExportableNVFP4Tensor) for w in (down, up, gate))
    assert down.dtype == torch.bfloat16
    assert float(down.per_tensor_scale) == 0.25
    assert float(up.per_tensor_scale) == 0.5
    assert float(gate.per_tensor_scale) == 1.0
    assert down.act_per_tensor_scale is None


@requires_gguf
def test_gguf_nvfp4_static_act_scale():
    tensors = [
        *TENSORS,
        ("blk.0.ffn_up.input_scale", scalar(0.003)),
        ("blk.0.ffn_gate.input_scale", scalar(0.004)),
    ]
    sd = load(tensors, static_act_scale=True)
    assert sd["layers.0.mlp.down_proj.weight"].act_per_tensor_scale == (
        pytest.approx(0.002)
    )
    assert sd["layers.0.mlp.gate_proj.weight"].act_per_tensor_scale == (
        pytest.approx(0.004)
    )
    # A static scale is requested but some weights have none.
    with pytest.raises(ValueError, match="without .input_scale"):
        load(TENSORS, static_act_scale=True)


@requires_gguf
def test_gguf_nvfp4_rejects_non_scalar_companion():
    tensors = [*TENSORS[1:], ("blk.0.ffn_down.scale", np.ones(2, dtype=np.float32))]
    with pytest.raises(ValueError, match="scalar F32 companion"):
        load(tensors)


@requires_gguf
def test_gguf_nvfp4_rejects_unpaired_companion():
    tensors = [*TENSORS, ("blk.0.attn_v.scale", scalar(0.5))]
    with pytest.raises(ValueError, match="blk.0.attn_v.scale"):
        load(tensors)


@pytest.mark.skipif(not HAS_GGUF, reason="needs gguf")
def test_gguf_without_nvfp4_type_still_loads():
    without_nvfp4 = IntEnum(
        "GGMLQuantizationType",
        {t.name: t.value for t in gguf.GGMLQuantizationType if t.name != "NVFP4"},
    )
    norm = np.ones(128, dtype=np.float32)
    with mock.patch.object(gguf, "GGMLQuantizationType", without_nvfp4):
        sd = load([("blk.0.ffn_norm.weight", norm)])
    assert list(sd) == ["layers.0.mlp.norm.weight"]
    assert torch.equal(sd["layers.0.mlp.norm.weight"].float(), torch.from_numpy(norm))


def exportable(w):
    return ExportableNVFP4Tensor(
        w.qdata.view(torch.uint32), w.scale, w.per_tensor_scale, 16, w.orig_dtype
    )


def shards(k):
    generator = torch.Generator().manual_seed(2026)
    result = []
    for n, global_scale in zip((31, 19, 47), (0.25, 0.37, 0.083)):
        q = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
        s = (
            (0.03 + torch.rand(n, k // 16, generator=generator) * 0.12)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        result.append(CudaNVFP4Tensor(q, s, torch.tensor([global_scale])))
    return result


def atomic_state_dict():
    sd = {}
    for layer in range(2):
        for name, w in zip(("q", "k", "og"), shards(192)):
            sd[f"layers.{layer}.self_attn.{name}_proj.weight"] = exportable(w)
        for name in ("gate", "up"):
            sd[f"layers.{layer}.mlp.{name}_proj.weight"] = exportable(shards(192)[0])
    return sd


def test_cuda_fuses_nvfp4_with_per_row_scales():
    sd = atomic_state_dict()
    fused, groups = _fuse_state_dict(sd, "cuda", 2)
    assert groups == {
        "layers.{}.self_attn.qko_proj.weight",
        "layers.{}.mlp.gate_up_proj.weight",
    }
    qko = fused["layers.0.self_attn.qko_proj.weight"]
    assert isinstance(qko, CudaNVFP4Tensor)
    assert qko.per_tensor_scale.numel() == 97
    assert fused["layers.0.mlp.gate_up_proj.weight"].per_tensor_scale.numel() == 1
    assert torch.equal(
        qko.dequantize(),
        torch.cat(
            [
                sd[f"layers.0.self_attn.{name}_proj.weight"].dequantize()
                for name in ("q", "k", "og")
            ]
        ),
    )
    mlx, mlx_groups = _fuse_state_dict(sd, "mlx", 2)
    assert mlx_groups == {"layers.{}.mlp.gate_up_proj.weight"}
    assert isinstance(mlx["layers.0.mlp.gate_up_proj.weight"], ExportableNVFP4Tensor)


def test_cuda_nvfp4_group_stays_separate_when_unfusable():
    sd = atomic_state_dict()
    sd["layers.1.self_attn.k_proj.weight"].act_per_tensor_scale = 0.001
    fused, groups = _fuse_state_dict(sd, "cuda", 2)
    assert groups == {"layers.{}.mlp.gate_up_proj.weight"}
    assert all(f"layers.{layer}.self_attn.q_proj.weight" in fused for layer in (0, 1))

    sd = atomic_state_dict()
    sd["layers.0.mlp.up_proj.weight"] = sd["layers.0.mlp.up_proj.weight"].dequantize()
    fused, groups = _fuse_state_dict(sd, "cuda", 2)
    assert groups == {"layers.{}.self_attn.qko_proj.weight"}
    assert fused["layers.0.mlp.up_proj.weight"] is sd["layers.0.mlp.up_proj.weight"]
