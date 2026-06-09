#!/usr/bin/env python3
"""
Plan A smoke test — validates the AOT half of the coopmat benchmark wiring
WITHOUT needing a GPU or a real model.

Checks:
  1. TensorRepSet.make_tensor_repr honors `prefer_storage` (the storage_type_override fix).
  2. A texture-only repset stays texture under a buffer preference (no crash / safe fallback).
  3. VulkanPartitioner injects storage_type_override=BUFFER when ET_VK_FORCE_BUFFER is set,
     and an explicit compile option always wins.
  4. A small multi-op graph (linear + layernorm + gelu + add) lowers end-to-end under
     global buffer with no crash (de-risks "some op lacks a buffer variant").

Run:  python yanwen/scripts/smoke_test_plan_a.py
Needs the editable venv (op_registry / utils edits must be live).
"""

import os

import torch
from torch.export import export

import executorch.backends.vulkan.utils as utils
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType
from executorch.exir import to_edge_transform_and_lower


def check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    assert ok, name


def test_make_tensor_repr():
    rs = utils.CONTIGUOUS_ANY  # both buffer and texture valid
    check("ANY default -> texture", rs.make_tensor_repr().storage_type, VkStorageType.TEXTURE_3D)
    check(
        "ANY prefer buffer -> buffer",
        rs.make_tensor_repr(VkStorageType.BUFFER).storage_type,
        VkStorageType.BUFFER,
    )
    tex = utils.WIDTH_PACKED_TEXTURE  # texture-only
    check(
        "texture-only prefer buffer -> stays texture (safe)",
        tex.make_tensor_repr(VkStorageType.BUFFER).storage_type,
        VkStorageType.TEXTURE_3D,
    )


def _specs(p):
    return {s.key: int.from_bytes(s.value, "little") for s in p.delegation_spec.compile_specs}

def test_partitioner_env():
    os.environ.pop("ET_VK_FORCE_BUFFER", None)
    check("no env -> no override", "storage_type_override" in _specs(VulkanPartitioner({})), False)

    os.environ["ET_VK_FORCE_BUFFER"] = "1"
    check(
        "ET_VK_FORCE_BUFFER=1 -> BUFFER",
        _specs(VulkanPartitioner({})).get("storage_type_override"),
        int(VkStorageType.BUFFER),
    )
    check(
        "explicit option wins over env",
        _specs(VulkanPartitioner({"storage_type_override": VkStorageType.TEXTURE_3D})).get(
            "storage_type_override"
        ),
        int(VkStorageType.TEXTURE_3D),
    )
    os.environ.pop("ET_VK_FORCE_BUFFER", None)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        h = self.act(self.fc1(self.ln(x)))
        return self.fc2(h) + x


def test_global_buffer_lower():
    os.environ["ET_VK_FORCE_BUFFER"] = "1"
    ep = export(_Tiny().eval(), (torch.randn(1, 32, 64),))
    to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner({})])
    print("[PASS] small multi-op graph lowered under global buffer, no crash")
    os.environ.pop("ET_VK_FORCE_BUFFER", None)


if __name__ == "__main__":
    test_make_tensor_repr()
    test_partitioner_env()
    test_global_buffer_lower()
    print("\nALL PLAN A SMOKE CHECKS PASSED")
