#!/usr/bin/env python3
"""Local validation for issue #20554 (no MLX runner / no full pip install)."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"


def _bootstrap_executorch() -> None:
    """Wire executorch.* to repo paths (Windows git symlinks are often broken)."""
    import importlib.util

    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(REPO))

    import executorch

    def _load_pkg(name: str) -> None:
        entry = SRC / "executorch" / name
        if entry.is_file():
            target = entry.read_text(encoding="utf-8").strip().replace("/", os.sep)
            real_root = (entry.parent / target).resolve()
        else:
            real_root = (REPO / name).resolve()
        init_py = real_root / "__init__.py"
        if not init_py.exists():
            return
        full_name = f"executorch.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            init_py,
            submodule_search_locations=[str(real_root)],
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        setattr(executorch, name, mod)
        spec.loader.exec_module(mod)

    def _load_subpkg(parent: str, name: str, root: Path) -> None:
        init_py = root / "__init__.py"
        if not init_py.exists():
            return
        full_name = f"{parent}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            init_py,
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        parent_mod = sys.modules[parent]
        setattr(parent_mod, name, mod)
        spec.loader.exec_module(mod)

    # exir.tracer needs executorch.extension.pytree (pybindings optional).
    ext = types.ModuleType("executorch.extension")
    sys.modules["executorch.extension"] = ext
    executorch.extension = ext
    _load_subpkg("executorch.extension", "pytree", REPO / "extension" / "pytree")

    _load_pkg("exir")
    _load_pkg("backends")


def _load_custom_ops():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mlx_custom_ops",
        REPO / "backends" / "mlx" / "custom_ops.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eager_moe_ops() -> None:
    import torch

    _load_custom_ops()

    torch.manual_seed(0)
    top_k = 2
    D, H = 32, 16

    # --- decode path (N=1, sort_cutoff=1) ---
    x = torch.randn(1, D)
    expert_indices = torch.tensor([[1, 3]], dtype=torch.int64)
    x_in, idx, sort_flag, inv = torch.ops.mlx.moe_gather_inputs(
        x, expert_indices, top_k, 1
    )
    assert x_in.shape == (2, 1, D)
    assert idx.shape == (2,)
    assert sort_flag.item() == 0
    assert inv.numel() == 0

    down = torch.randn(2, 1, H)
    out = torch.ops.mlx.moe_scatter_outputs(down, sort_flag, inv, top_k)
    assert out.shape == (1, top_k, H)
    assert torch.allclose(out, down.squeeze(-2).reshape(1, top_k, H).clone())

    # --- prefill path (N=4, sort_cutoff=1) ---
    x = torch.randn(4, D)
    expert_indices = torch.tensor([[2, 0], [1, 3], [0, 2], [3, 1]], dtype=torch.int64)
    x_in, idx, sort_flag, inv = torch.ops.mlx.moe_gather_inputs(
        x, expert_indices, top_k, 1
    )
    assert x_in.shape == (8, 1, D)
    assert sort_flag.item() == 1
    assert inv.shape == (8,)

    # Reference sorted path (mirror eager in custom_ops)
    flat = expert_indices.flatten()
    order = flat.argsort().to(torch.int32)
    inv_ref = order.argsort().to(torch.int32)
    idx_ref = flat[order].to(torch.int32)
    x_ref = x[(order // top_k).to(torch.int64)].unsqueeze(-2)
    assert torch.equal(idx, idx_ref)
    assert torch.equal(inv, inv_ref)
    assert torch.allclose(x_in, x_ref)

    down = torch.randn(8, 1, H)
    out = torch.ops.mlx.moe_scatter_outputs(down, sort_flag, inv, top_k)
    down_sq = down.squeeze(-2)
    ref = down_sq[inv_ref].reshape(4, top_k, H)
    assert torch.allclose(out, ref)

    print("PASS: eager moe_gather_inputs / moe_scatter_outputs")


def test_opcheck() -> None:
    import torch
    from torch.library import opcheck

    _load_custom_ops()

    x = torch.randn(4, 32)
    expert_indices = torch.randint(0, 8, (4, 2))
    opcheck(torch.ops.mlx.moe_gather_inputs, (x, expert_indices, 2, 1))
    down = torch.randn(8, 1, 16)
    sort_experts = torch.tensor(1, dtype=torch.int32)
    inv_order = torch.arange(8, dtype=torch.int32)
    opcheck(
        torch.ops.mlx.moe_scatter_outputs,
        (down, sort_experts, inv_order, 2),
    )
    print("PASS: torch.library.opcheck")


def test_export_traces_moe_ops() -> None:
    import torch
    import torch.nn as nn
    from torch.export import export

    _load_custom_ops()

    class MoeModel(nn.Module):
        def forward(self, x, expert_indices):
            gathered = torch.ops.mlx.moe_gather_inputs(x, expert_indices, 2, 1)
            return torch.ops.mlx.moe_scatter_outputs(
                torch.randn(gathered[0].shape[0], 1, 16),
                gathered[2],
                gathered[3],
                2,
            )

    ep = export(
        MoeModel(),
        (torch.randn(4, 32), torch.randint(0, 4, (4, 2))),
    )
    targets = {
        str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_function"
    }
    assert any("moe_gather_inputs" in t for t in targets), targets
    assert any("moe_scatter_outputs" in t for t in targets), targets
    print("PASS: torch.export traces moe ops as leaf nodes")


def _count_mlx_nodes(mlx_graph) -> dict[str, int]:
    from collections import Counter

    return dict(
        Counter(
            type(instr.op).__name__
            for chain in mlx_graph.instruction_chains
            for instr in chain.instructions
        )
    )


def test_export_lowering_node_counts() -> None:
    import torch
    import torch.nn as nn
    from torch.export import export

    from executorch.backends.mlx import custom_ops  # noqa: F401
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
    from executorch.exir import EdgeCompileConfig, to_edge

    class MoeGather(nn.Module):
        def forward(self, x, expert_indices):
            return torch.ops.mlx.moe_gather_inputs(x, expert_indices, 2, 1)[0]

    cfg = EdgeCompileConfig(_check_ir_validity=False)

    ep = export(MoeGather(), (torch.randn(4, 32), torch.randint(0, 4, (4, 2))))
    prefill = _count_mlx_nodes(
        MLXProgramBuilder(to_edge(ep, compile_config=cfg).exported_program()).build()
    )
    assert prefill.get("IfNode", 0) == 0, prefill
    assert prefill.get("ArgsortNode", 0) == 2, prefill
    assert prefill.get("RepeatNode", 0) == 0, prefill
    print(f"PASS: MLX lowering (prefill): {prefill}")

    ep1 = export(MoeGather(), (torch.randn(1, 32), torch.randint(0, 4, (1, 2))))
    decode = _count_mlx_nodes(
        MLXProgramBuilder(to_edge(ep1, compile_config=cfg).exported_program()).build()
    )
    assert decode.get("IfNode", 0) == 0, decode
    assert decode.get("ArgsortNode", 0) == 0, decode
    assert decode.get("RepeatNode", 0) == 1, decode
    print(f"PASS: MLX lowering (decode): {decode}")


def test_switch_mlp_forward() -> None:
    import torch
    import torch.nn as nn
    from executorch.backends.mlx import custom_ops  # noqa: F401
    from executorch.backends.mlx.llm.switch import SwitchMLP, pack_all_switch_linears

    mlp = SwitchMLP(32, 64, num_experts=4, sort_cutoff=1)
    for mod in mlp.modules():
        if hasattr(mod, "experts"):
            for e in mod.experts:
                nn.init.uniform_(e.weight, -0.1, 0.1)
    pack_all_switch_linears(mlp)

    x = torch.randn(4, 32)
    weights = torch.softmax(torch.randn(4, 2), dim=-1)
    indices = torch.randint(0, 4, (4, 2))

    out_prefill = mlp(x, weights, indices, top_k=2)
    assert out_prefill.shape == (4, 32)

    x1 = torch.randn(1, 32)
    w1 = torch.softmax(torch.randn(1, 2), dim=-1)
    i1 = torch.randint(0, 4, (1, 2))
    out_decode = mlp(x1, w1, i1, top_k=2)
    assert out_decode.shape == (1, 32)
    print("PASS: SwitchMLP forward (prefill + decode)")


def main() -> int:
    test_eager_moe_ops()
    test_opcheck()
    test_export_traces_moe_ops()
    _bootstrap_executorch()
    test_switch_mlp_forward()
    try:
        test_export_lowering_node_counts()
    except Exception as e:
        print(f"FAIL: MLX lowering: {e}")
        return 1
    print("\nAll validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
