# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke-test export for PortableBackend_v2.

Exports a tiny PTE that targets the new "PortableBackend_v2" backend
(registered by libportable_backend_v2.a). Reuses the existing
PortablePartitioner machinery; just overrides the delegation spec's
backend_id to point at the v2 backend.
"""

import os
import sys

# Make the in-repo executorch package importable by adding the parent of
# the executorch/ root to sys.path.
_EXECUTORCH_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_PARENT = os.path.dirname(_EXECUTORCH_ROOT)
for _p in (_PARENT, _EXECUTORCH_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.export import export

from executorch.exir import to_edge, EdgeCompileConfig
from executorch.exir.backend.partitioner import DelegationSpec
from executorch.backends.portable.partitioner import PortablePartitioner
# Importing this triggers the Python-side BackendDetails registration
# under the name "PortableBackend_v2" (matched at runtime by the C++
# register_backend call in runtime_v2/PortableBackend_v2.cpp).
from executorch.backends.portable.preprocess_v2 import PortableBackend_v2  # noqa: F401


def _patch_partitioner_to_v2(p: PortablePartitioner) -> PortablePartitioner:
    """Mutate p's delegation_spec so the backend_id points at v2."""
    old = p.delegation_spec
    p.delegation_spec = DelegationSpec("PortableBackend_v2", old.compile_specs)
    return p


def export_v2(model, example_inputs, name: str, dynamic_shapes=None) -> str:
    print(f"\nExporting {name} (delegated to PortableBackend_v2)...")
    if dynamic_shapes is not None:
        exported = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    else:
        exported = export(model, example_inputs)
    # Skip dim_order so .clone() exports as aten::clone (which we
    # eventually want in our op registry) rather than
    # dim_order_ops::_clone_dim_order (a dim-order-aware variant our
    # v2 op registry doesn't dispatch).
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_skip_dim_order=True))

    partitioner = _patch_partitioner_to_v2(PortablePartitioner())
    delegated = edge.to_backend(partitioner)
    et_program = delegated.to_executorch()

    path = f"/tmp/{name}_v2.pte"
    with open(path, "wb") as f:
        f.write(et_program.buffer)
    print(f"  Saved to {path} ({len(et_program.buffer)} bytes)")
    return path


def main() -> None:
    print("=" * 60)
    print("PortableBackend_v2 — smoke-test export")
    print("=" * 60)

    # Tiny add: y = x + x
    class TinyAdd(torch.nn.Module):
        def forward(self, x):
            return x + x

    export_v2(TinyAdd().eval(), (torch.ones(2, 3),), "tiny_add")

    # Add with constant: y = x + w  (exercises NDM upload_constant path)
    class AddConst(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.full((2, 3), 0.5))

        def forward(self, x):
            return x + self.w

    export_v2(AddConst().eval(), (torch.ones(2, 3),), "add_const")

    # Two inputs, one output:
    #   t1 = x + y         intermediate (mem_obj_id)
    #   t2 = t1 + x        intermediate (may share with t1)
    #   out = t2 - y
    # x = ones(2,3), y = ones(2,3)*2  → out = (1+2)+1 - 2 = 2
    class TwoInputsChain(torch.nn.Module):
        def forward(self, x, y):
            t1 = x + y
            t2 = t1 + x
            out = t2 - y
            return out

    export_v2(
        TwoInputsChain().eval(),
        (torch.ones(2, 3), torch.full((2, 3), 2.0)),
        "two_inputs_chain",
    )

    # Many sequential adds — exercises mem_obj_id sharing aggressively.
    # Use tensor+tensor so the loop doesn't fold to a single op.
    # Result: x + 10*y = 0 + 10*1 = 10 everywhere.
    class ManyAdds(torch.nn.Module):
        def forward(self, x, y):
            for _ in range(10):
                x = x + y
            return x

    export_v2(
        ManyAdds().eval(),
        (torch.zeros(4, 4), torch.ones(4, 4)),
        "many_adds",
    )

    # Linear: matmul + bias.
    # x: [3,4]; W: [5,4]; bias: [5].  out = x @ W^T + bias  → [3,5]
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((5, 4), 0.25))
            self.bias = torch.nn.Parameter(torch.full((5,), -1.0))

        def forward(self, x):
            return torch.mm(x, self.weight.t()) + self.bias

    export_v2(Linear().eval(), (torch.ones(3, 4),), "linear")

    # Dynamic shape: x has a dynamic batch dim with bound min=1, max=8.
    # At export time we trace with batch=3, but the model should run with
    # any batch in [1, 8]. AOT memory planning sizes intermediates to
    # max_shape (batch=8); per-execute, the actual batch is propagated
    # via TensorImpl.sizes().
    class DynBatch(torch.nn.Module):
        def forward(self, x, y):
            t1 = x + y
            t2 = t1 * y
            return t2 - x

    batch = torch.export.Dim("batch", min=1, max=8)
    export_v2(
        DynBatch().eval(),
        (torch.ones(3, 4), torch.full((3, 4), 2.0)),
        "dyn_batch",
        dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
    )

    # Dynamic shape across a cross-runtime boundary:
    #   permute_copy(weight) → CPU (not in MetalOpRegistry)
    #   mm(x, perm_weight) → Metal (TransferStep cpu→metal for perm_weight)
    #   add(mm_out, bias) → Metal
    # The mm output's shape depends on x's dynamic batch dim;
    # the add output's bias broadcast shape varies too. This exercises
    # transfer_tensor's resize_tensor shape propagation under dynamism.
    class DynLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((5, 4), 0.25))
            self.bias = torch.nn.Parameter(torch.full((5,), -1.0))

        def forward(self, x):
            return torch.mm(x, self.weight.t()) + self.bias

    dyn_batch_lin = torch.export.Dim("dyn_batch_lin", min=1, max=8)
    export_v2(
        DynLinear().eval(),
        (torch.ones(3, 4),),
        "dyn_linear",
        dynamic_shapes={"x": {0: dyn_batch_lin}},
    )

    # Stateful model with a mutable buffer (KV-cache pattern).
    # Each forward call adds x to the running accumulator (acc += x)
    # and returns the new accumulator value. State must persist across
    # execute() calls.
    #   call 1 with ones(2,3): output = [1,1,1; 1,1,1]
    #   call 2 with ones(2,3): output = [2,2,2; 2,2,2]
    #   call 3 with ones(2,3): output = [3,3,3; 3,3,3]
    class StatefulAdd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", torch.zeros(2, 3))

        def forward(self, x):
            self.acc.add_(x)
            return self.acc.clone()

    export_v2(StatefulAdd().eval(), (torch.ones(2, 3),), "stateful_add")

    print("\nDone. Run with:")
    for name in [
        "tiny_add",
        "add_const",
        "two_inputs_chain",
        "many_adds",
        "linear",
        "dyn_batch",
        "dyn_linear",
        "stateful_add",
    ]:
        print(f"  ./cmake-out/executor_runner --model_path /tmp/{name}_v2.pte")


if __name__ == "__main__":
    main()
