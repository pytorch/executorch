# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side checks for sharing one Ethos-U weight blob across per-window methods.

M0: two input windows produce byte-identical Vela weight_data, differing cmd_data.
M1: a two-method program stores the shared weights once, in an external .ptd.
M2a: with place_named_data_inline, the shared weights are stored once inside a
self-contained .pte.

FVP execution of both methods lives in tester/fb/test/test_multi_window_fvp.py.
"""

import hashlib
import json
import os
import tempfile

import torch
from executorch.backends.arm import arm_vela
from executorch.backends.arm.ethosu import backend as ethosu_backend
from executorch.backends.arm.ethosu import (
    EthosUCompileSpec,
    EthosUPartitioner,
    VelaExternalBlockPlacements,
)
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize.padding import aligned_size
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TimeConv(torch.nn.Module):
    """Conv stack over the time axis; weights are independent of window length."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = torch.nn.Conv2d(8, 16, (3, 1), padding=(1, 0))
        self.c2 = torch.nn.Conv2d(16, 16, (3, 1), padding=(1, 0))
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.c2(self.relu(self.c1(x))))


def _window_input(time: int, fill_random: bool = False) -> torch.Tensor:
    shape = (1, 8, time, 1)
    return torch.randn(*shape) if fill_random else torch.zeros(*shape)


def _parse_blocks(binary: bytes) -> dict[str, bytes]:
    blocks: dict[str, bytes] = {}
    offset = 0
    while offset < len(binary):
        name, size, _external, _reserved = arm_vela._BLOCK_HEADER.unpack_from(
            binary, offset
        )
        start = offset + arm_vela._BLOCK_HEADER.size
        blocks[name.rstrip(b"\x00").decode()] = binary[start : start + size]
        offset = start + aligned_size(size, arm_vela._BLOCK_ALIGNMENT)
    return blocks


def _quantize_once(model: torch.nn.Module) -> torch.nn.Module:
    quantizer = EthosUQuantizer(common.get_u55_compile_spec())
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    calib = _window_input(32, fill_random=True)
    captured = export(model, (calib,), strict=True).module()
    prepared = prepare_pt2e(captured, quantizer)  # type: ignore[arg-type]
    prepared(calib)
    converted = convert_pt2e(prepared, fold_quantize=True)
    DuplicateDynamicQuantChainPass()(converted)
    return converted


def _lower_and_capture(converted: torch.nn.Module, time: int) -> list[dict[str, bytes]]:
    """Lower the quantized model at a given window and return per-delegate blocks."""
    spec = common.get_u55_compile_spec()
    ep = export(converted, (_window_input(time),), strict=True)

    captured: list[bytes] = []
    real_vela = ethosu_backend.vela_compile

    def capturing_vela(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real_vela(*args, **kwargs)
        captured.append(res.processed_bytes)
        return res

    ethosu_backend.vela_compile = capturing_vela
    try:
        to_edge_transform_and_lower(ep, partitioner=[EthosUPartitioner(spec)])
    finally:
        ethosu_backend.vela_compile = real_vela

    return [_parse_blocks(b) for b in captured]


def test_two_windows_share_weight_data() -> None:
    torch.manual_seed(0)
    model = TimeConv().eval()
    converted = _quantize_once(model)

    a = _lower_and_capture(converted, 32)
    b = _lower_and_capture(converted, 64)

    summary = {
        "num_delegates_w32": len(a),
        "num_delegates_w64": len(b),
        "delegates": [],
    }
    for i, (da, db) in enumerate(zip(a, b)):
        wa, wb = da.get("weight_data", b""), db.get("weight_data", b"")
        ca, cb = da.get("cmd_data", b""), db.get("cmd_data", b"")
        summary["delegates"].append(
            {
                "delegate": i,
                "weight_len_w32": len(wa),
                "weight_len_w64": len(wb),
                "weight_sha_w32": hashlib.sha256(wa).hexdigest()[:16],
                "weight_sha_w64": hashlib.sha256(wb).hexdigest()[:16],
                "weight_same": wa == wb,
                "cmd_len_w32": len(ca),
                "cmd_len_w64": len(cb),
                "cmd_same": ca == cb,
            }
        )
    print("M0_WEIGHT_SHARE " + json.dumps(summary))
    with open("/tmp/m0_weight_share.json", "w") as f:
        json.dump(summary, f, indent=2)

    assert len(a) == len(b) and len(a) >= 1, f"delegate count mismatch: {len(a)} vs {len(b)}"
    for d in summary["delegates"]:
        assert d["weight_len_w32"] > 0, f"delegate {d['delegate']} has empty weights"
        assert d["weight_same"], (
            f"delegate {d['delegate']}: weight_data differs across windows "
            f"({d['weight_sha_w32']} vs {d['weight_sha_w64']})"
        )
        assert not d["cmd_same"], (
            f"delegate {d['delegate']}: cmd_data identical across windows "
            "(expected to differ)"
        )


_WEIGHT_TAG = "ethos_shared_weights"


def _u55_spec_with_external_weights() -> EthosUCompileSpec:
    """U55 spec matching common.get_u55_compile_spec, but with weight_data emitted
    as an external, content-addressed named-data block for cross-method sharing."""
    return EthosUCompileSpec(
        "ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--arena-cache-size=2097152"],
        config_ini="Arm/vela.ini",
        max_scratch_size=2 * 1024 * 1024,
        external_block_placements=VelaExternalBlockPlacements(weight_data=_WEIGHT_TAG),
    )


def test_two_methods_share_one_weight_buffer() -> None:
    """Build a single program with a method per window, weights externalized, and
    confirm both methods dedup to exactly one shared weight buffer."""
    torch.manual_seed(0)
    converted = _quantize_once(TimeConv().eval())

    ep32 = export(converted, (_window_input(32),), strict=True)
    ep64 = export(converted, (_window_input(64),), strict=True)

    spec = _u55_spec_with_external_weights()
    edge = to_edge_transform_and_lower(
        {"forward_w32": ep32, "forward_w64": ep64},
        partitioner=[EthosUPartitioner(spec)],
    )

    ndso = edge._named_data_store.get_named_data_store_output()
    weight_entries = ndso.external_data.get(_WEIGHT_TAG, {})
    unique_weight_buffers = {
        entry.buffer_index for entry in weight_entries.values()
    }
    weight_bytes = (
        ndso.buffers[next(iter(unique_weight_buffers))]
        if unique_weight_buffers
        else b""
    )

    methods = list(getattr(edge, "_edge_programs", {}).keys())

    # Serialize to the real artifacts (.pte + external .ptd) and count how many
    # times the shared weight payload physically appears on disk.
    et = edge.to_executorch()
    with tempfile.TemporaryDirectory(prefix="m1_pte_") as outdir:
        pte_path = os.path.join(outdir, "model.pte")
        et.save(pte_path)
        et.write_tensor_data_to_file(outdir)
        with open(pte_path, "rb") as f:
            pte_data = f.read()
        ptd_data = b""
        ptd_files = []
        for name in sorted(os.listdir(outdir)):
            if name.endswith(".ptd"):
                ptd_files.append(name)
                with open(os.path.join(outdir, name), "rb") as f:
                    ptd_data += f.read()
        occurrences_in_pte = (
            pte_data.count(weight_bytes) if weight_bytes else -1
        )
        occurrences_in_ptd = (
            ptd_data.count(weight_bytes) if weight_bytes else -1
        )

    summary = {
        "methods": methods,
        "num_weight_keys": len(weight_entries),
        "num_unique_weight_buffers": len(unique_weight_buffers),
        "weight_keys": list(weight_entries.keys()),
        "weight_buffer_len": len(weight_bytes),
        "ptd_files": ptd_files,
        "pte_size": len(pte_data),
        "weight_occurrences_in_pte": occurrences_in_pte,
        "weight_occurrences_in_ptd": occurrences_in_ptd,
    }
    print("M1_WEIGHT_DEDUP " + json.dumps(summary))
    with open("/tmp/m1_weight_dedup.json", "w") as f:
        json.dump(summary, f, indent=2)

    assert len(methods) == 2, f"expected 2 methods, got {methods}"
    assert len(weight_entries) == 1, (
        f"expected 1 shared weight key across both methods, got {len(weight_entries)}"
    )
    assert len(unique_weight_buffers) == 1, (
        "expected both methods to reference one shared weight buffer, "
        f"got {len(unique_weight_buffers)}"
    )
    assert len(weight_bytes) > 0, "shared weight buffer is empty"
    # Artifact-level proof: the weight payload is stored exactly once, in the PTD,
    # and is not duplicated per method anywhere in the serialized output.
    assert occurrences_in_ptd == 1, (
        f"weight payload appears {occurrences_in_ptd}x in .ptd (expected exactly 1)"
    )
    assert occurrences_in_pte == 0, (
        f"weight payload appears {occurrences_in_pte}x in .pte "
        "(expected 0; weights are externalized to the .ptd)"
    )


def _u55_spec_inline_shared_weights() -> EthosUCompileSpec:
    """Like _u55_spec_with_external_weights, but the shared weight block is
    deduplicated inline in the .pte (PteDataMap-resolved on device) instead of a
    separate .ptd -- yielding one self-contained .pte for multiple windows."""
    return EthosUCompileSpec(
        "ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--arena-cache-size=2097152"],
        config_ini="Arm/vela.ini",
        max_scratch_size=2 * 1024 * 1024,
        external_block_placements=VelaExternalBlockPlacements(weight_data=_WEIGHT_TAG),
        place_named_data_inline=True,
    )


def test_two_methods_share_one_inline_weight_buffer() -> None:
    """Build one self-contained .pte with a method per window and weights shared
    inline (no .ptd); confirm the weight payload is stored exactly once in the
    .pte and no external data file is produced."""
    torch.manual_seed(0)
    converted = _quantize_once(TimeConv().eval())

    ep32 = export(converted, (_window_input(32),), strict=True)
    ep64 = export(converted, (_window_input(64),), strict=True)

    spec = _u55_spec_inline_shared_weights()
    edge = to_edge_transform_and_lower(
        {"forward_w32": ep32, "forward_w64": ep64},
        partitioner=[EthosUPartitioner(spec)],
    )

    ndso = edge._named_data_store.get_named_data_store_output()
    pte_entries = ndso.pte_data
    unique_buffers = {entry.buffer_index for entry in pte_entries.values()}
    weight_bytes = (
        ndso.buffers[next(iter(unique_buffers))] if unique_buffers else b""
    )
    methods = list(getattr(edge, "_edge_programs", {}).keys())

    et = edge.to_executorch()
    with tempfile.TemporaryDirectory(prefix="m2a_pte_") as outdir:
        pte_path = os.path.join(outdir, "model.pte")
        et.save(pte_path)
        et.write_tensor_data_to_file(outdir)
        with open(pte_path, "rb") as f:
            pte_data = f.read()
        ptd_files = [n for n in os.listdir(outdir) if n.endswith(".ptd")]
        occurrences_in_pte = pte_data.count(weight_bytes) if weight_bytes else -1

    summary = {
        "methods": methods,
        "num_external_tags": len(ndso.external_data),
        "num_pte_named_keys": len(pte_entries),
        "num_unique_inline_buffers": len(unique_buffers),
        "weight_buffer_len": len(weight_bytes),
        "ptd_files": ptd_files,
        "pte_size": len(pte_data),
        "weight_occurrences_in_pte": occurrences_in_pte,
    }
    print("M2A_INLINE_SHARE " + json.dumps(summary))
    with open("/tmp/m2a_inline_share.json", "w") as f:
        json.dump(summary, f, indent=2)

    assert len(methods) == 2, f"expected 2 methods, got {methods}"
    assert ndso.external_data == {}, "expected no external .ptd data (inline mode)"
    assert ptd_files == [], f"expected no .ptd files, got {ptd_files}"
    assert len(unique_buffers) == 1, (
        f"expected 1 shared inline weight buffer, got {len(unique_buffers)}"
    )
    assert len(weight_bytes) > 0, "shared inline weight buffer is empty"
    assert occurrences_in_pte == 1, (
        f"weight payload appears {occurrences_in_pte}x in the .pte (expected 1)"
    )
