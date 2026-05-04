"""
Export one .pte per in-place op test case for the C++ test_inplace_native driver.

Each .pte is a tiny model that:
  * Takes input tensor(s).
  * Performs one (or several) in-place op(s).
  * Returns the mutated tensor(s).

The export bypasses the partitioner — we call ET's reinplace_pass directly
on the edge program so the in-place ops land at the TOP level of the .pte
(not inside a delegate's processed bytes). This lets the C++ driver load
each .pte directly via NativeBackend::init() without going through the
partitioner machinery.

Outputs (all in /tmp):
  inplace_unary_chain.pte   ← chain of 13 unary in-place ops in one program
  inplace_add_Tensor.pte
  inplace_sub_Tensor.pte
  inplace_mul_Tensor.pte
  inplace_div_Tensor.pte
  inplace_clamp_.pte        ← unary with bake-in scalar bounds
  (index/scatter pte's added when reinplace supports them)

The C++ driver loads each + verifies output element-wise against an
expected vector hardcoded based on the model's eager output.
"""

import sys
import torch
from executorch.exir import to_edge, EdgeCompileConfig
from torch.export import export
from executorch.backends.native.preprocess import BACKEND_INPLACE_OPS


def export_inplace(model, example, name: str) -> str:
    """Export `model` with reinplace_pass applied, save to /tmp/inplace_<name>.pte."""
    print(f"\n=== Exporting in-place op: {name} ===")
    ep = export(model, example)
    edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))

    from executorch.exir.passes.reinplace import reinplace_pass
    from executorch.exir.passes import SpecPropPass
    from executorch.backends.native.preprocess import _apply_passes

    inner = edge.exported_program()
    inner = _apply_passes(inner, [
        lambda p: reinplace_pass(p, ops_to_inplace=BACKEND_INPLACE_OPS),
        SpecPropPass(),
    ])
    edge._edge_programs["forward"] = inner

    et = edge.to_executorch()
    path = f"/tmp/inplace_{name}.pte"
    with open(path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {path} ({len(et.buffer)} bytes)")

    prog = et._emitter_output.program
    for ep_i, plan in enumerate(prog.execution_plan):
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")

    # Print eager-mode expected so we can hardcode it in the C++ test.
    with torch.no_grad():
        out = model(*[t.clone() for t in example])
    print(f"  Eager output: {out.flatten().tolist()}")
    return path


def export_inplace_with_buffers(
    model, example, name: str, init_buffer_patterns=None
) -> str:
    """Like export_inplace but also handles registered mutable buffers:
    runs InitializedMutableBufferPass + insert_write_back_for_buffers_pass
    so the buffer's initial state is serialized into the .pte and a
    writeback copy_ is emitted for its mutation."""
    print(f"\n=== Exporting in-place op (with buffer): {name} ===")
    ep = export(model, example)
    edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))

    from executorch.exir.passes.reinplace import reinplace_pass
    from executorch.exir.passes import SpecPropPass
    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
    from executorch.exir.passes.insert_write_back_for_buffers_pass import (
        insert_write_back_for_buffers_pass,
    )
    from torch.export.graph_signature import OutputKind
    from executorch.backends.native.preprocess import _apply_passes

    inner = edge.exported_program()

    # Mark named buffers as having serialized initial state so they get
    # zero (or whatever) startup values rather than uninitialized memory.
    if init_buffer_patterns:
        inner = _apply_passes(
            inner, [InitializedMutableBufferPass(init_buffer_patterns)]
        )

    inner = _apply_passes(inner, [SpecPropPass()])
    inner = _apply_passes(inner, [
        lambda p: reinplace_pass(p, ops_to_inplace=BACKEND_INPLACE_OPS),
        SpecPropPass(),
    ])

    has_buf_mut = any(
        o.kind == OutputKind.BUFFER_MUTATION
        for o in inner.graph_signature.output_specs
    )
    if has_buf_mut:
        gm, sig = insert_write_back_for_buffers_pass(inner)
        inner._graph_module = gm
        inner._graph_signature = sig
        inner = _apply_passes(inner, [SpecPropPass()])

    edge._edge_programs["forward"] = inner
    et = edge.to_executorch()
    path = f"/tmp/inplace_{name}.pte"
    with open(path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {path} ({len(et.buffer)} bytes)")

    prog = et._emitter_output.program
    for ep_i, plan in enumerate(prog.execution_plan):
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")

    with torch.no_grad():
        out = model(*[t.clone() for t in example])
    print(f"  Eager output sum: {out.sum().item()}")
    return path


# ---------------- Unary chain --------------------------------------------
# Chains a sequence of unary in-place ops covering 13 different op kinds.
# The chain is constructed so the final value is computable cleanly:
#   x = [4, 9, 16, 25, 36, 49]
#   sqrt_     -> [2, 3, 4, 5, 6, 7]
#   neg_      -> [-2, -3, -4, -5, -6, -7]
#   abs_      -> [2, 3, 4, 5, 6, 7]
#   square_   -> [4, 9, 16, 25, 36, 49]
#   sqrt_     -> [2, 3, 4, 5, 6, 7]
#   relu_     -> [2, 3, 4, 5, 6, 7]
#   ceil_     -> [2, 3, 4, 5, 6, 7]
#   floor_    -> [2, 3, 4, 5, 6, 7]
#   round_    -> [2, 3, 4, 5, 6, 7]
#   trunc_    -> [2, 3, 4, 5, 6, 7]
#   relu6_    -> [2, 3, 4, 5, 6, 6]   (clip to 6)
#   hardtanh_ -> [2, 3, 4, 5, 5, 5]   (clip to [-3, 5])
#   neg_      -> [-2, -3, -4, -5, -5, -5]
#   relu_     -> [0, 0, 0, 0, 0, 0]
#   exp_      -> [1, 1, 1, 1, 1, 1]
#   log_      -> [0, 0, 0, 0, 0, 0]
class UnaryChain(torch.nn.Module):
    def forward(self, x):
        x.sqrt_()
        x.neg_()
        x.abs_()
        x.square_()
        x.sqrt_()
        x.relu_()
        x.ceil_()
        x.floor_()
        x.round_()
        x.trunc_()
        torch.nn.functional.relu6(x, inplace=True)
        torch.nn.functional.hardtanh_(x, -3.0, 5.0)
        x.neg_()
        x.relu_()
        x.exp_()
        x.log_()
        return x


class AddInplace(torch.nn.Module):
    def forward(self, x, y):
        x.add_(y)
        return x


class SubInplace(torch.nn.Module):
    def forward(self, x, y):
        x.sub_(y)
        return x


class MulInplace(torch.nn.Module):
    def forward(self, x, y):
        x.mul_(y)
        return x


class DivInplace(torch.nn.Module):
    def forward(self, x, y):
        x.div_(y)
        return x


class ClampInplace(torch.nn.Module):
    def forward(self, x):
        x.clamp_(-1.0, 1.0)
        return x


class IndexPutInplace(torch.nn.Module):
    """Writes `values` into `self` at `indices` (in-place).
    Tests aten.index_put → aten.index_put_ rewrite.
    Schema: index_put_(self, indices: Tensor?[], values, accumulate=False)
    """
    def forward(self, x, idx, vals):
        # x[idx] = vals — single 1D-index slice along dim 0.
        x.index_put_([idx], vals)
        return x


class IndexAddInplace(torch.nn.Module):
    """index_add_(self, dim, index, source, alpha=1):
    self[index[i]] += alpha * source[i] along dim.
    """
    def forward(self, x, idx, src):
        x.index_add_(0, idx, src)
        return x


class ScatterAddInplace(torch.nn.Module):
    """scatter_add_(self, dim, index, src):
    self[index[i,j], j] += src[i,j] for dim=0.
    """
    def forward(self, x, idx, src):
        x.scatter_add_(0, idx, src)
        return x


class MaskedScatterInplace(torch.nn.Module):
    """masked_scatter_(self, mask, source):
    For each True position in mask, write next element of source into self.
    """
    def forward(self, x, mask, src):
        x.masked_scatter_(mask, src)
        return x


class IndexCopyCache(torch.nn.Module):
    """HF-style KV-cache update: index_copy_ along the seq dimension.
    This is the canonical pattern HF transformers use for KV-cache
    insertion at runtime positions.
    """
    def __init__(self, max_len=8, n_heads=2, head_dim=4):
        super().__init__()
        self.register_buffer("k_cache",
                             torch.zeros(1, n_heads, max_len, head_dim))

    def forward(self, k_val, cache_position):
        self.k_cache.index_copy_(2, cache_position, k_val)
        return self.k_cache.clone()  # clone() avoids inline-constant from `+ 0`


def main() -> None:
    f23 = (2, 3)

    export_inplace(
        UnaryChain().eval(),
        (torch.tensor([[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]]),),
        "unary_chain",
    )

    export_inplace(
        AddInplace().eval(),
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
         torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])),
        "add_Tensor",
    )

    export_inplace(
        SubInplace().eval(),
        (torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
         torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        "sub_Tensor",
    )

    export_inplace(
        MulInplace().eval(),
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
         torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])),
        "mul_Tensor",
    )

    export_inplace(
        DivInplace().eval(),
        (torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
         torch.tensor([[2.0, 4.0, 5.0], [8.0, 10.0, 12.0]])),
        "div_Tensor",
    )

    export_inplace(
        ClampInplace().eval(),
        (torch.tensor([[-3.0, -0.5, 0.0], [0.5, 2.5, 5.0]]),),
        "clamp_",
    )

    # index_put_: x[idx] = vals (in-place).
    # x = zeros(2, 3), idx = LongTensor([0]), vals = [[1, 2, 3]]
    # → x[0, :] = [1, 2, 3] → expected [[1, 2, 3], [0, 0, 0]]
    export_inplace(
        IndexPutInplace().eval(),
        (
            torch.zeros(2, 3),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 2.0, 3.0]]),
        ),
        "index_put_",
    )

    # index_add_: x[idx[i], :] += src[i, :] along dim=0.
    # x = zeros(2, 3), idx = LongTensor([0, 1]),
    # src = [[1, 1, 1], [2, 2, 2]]
    # → x[0] += [1,1,1], x[1] += [2,2,2] → [[1,1,1],[2,2,2]]
    export_inplace(
        IndexAddInplace().eval(),
        (
            torch.zeros(2, 3),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        ),
        "index_add_",
    )

    # scatter_add_: x[idx[i,j], j] += src[i,j] for dim=0.
    # x = zeros(2, 3),
    # idx = [[0, 1, 0], [1, 0, 1]] (Long 2x3),
    # src = [[1, 1, 1], [2, 2, 2]] (Float 2x3).
    # Result: x = [[1, 2, 1], [2, 1, 2]].
    export_inplace(
        ScatterAddInplace().eval(),
        (
            torch.zeros(2, 3),
            torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long),
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        ),
        "scatter_add_",
    )

    # masked_scatter_: write next-source-element to each True position.
    # x = zeros(2, 3),
    # mask = [[True, False, True], [False, True, False]] (Bool 2x3),
    # source = [10, 20, 30, ...] (Float 1D ≥ 3 elems).
    # 3 True positions: (0,0), (0,2), (1,1) → [10, 0, 20, 0, 30, 0]
    export_inplace(
        MaskedScatterInplace().eval(),
        (
            torch.zeros(2, 3),
            torch.tensor([[True, False, True], [False, True, False]]),
            torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0]),
        ),
        "masked_scatter_",
    )

    # KV-cache index_copy_: HF-style per-token cache insertion.
    # k_cache shape [1, n_heads=2, max_len=8, head_dim=4].
    # k_val shape [1, 2, 2, 4] (2 new tokens), cache_position [3, 4].
    # → Writes k_val[..., i, :] into k_cache[..., positions[i], :].
    export_inplace_with_buffers(
        IndexCopyCache(max_len=8, n_heads=2, head_dim=4).eval(),
        (
            torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4),
            torch.tensor([3, 4], dtype=torch.long),
        ),
        "kvcache_index_copy",
        init_buffer_patterns=["k_cache"],
    )


if __name__ == "__main__":
    main()
