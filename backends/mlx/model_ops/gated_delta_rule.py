#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Gated delta rule custom op and pattern handler for MLX backend.

This module defines:
1. mlx::gated_delta_rule custom op with mutates_args=("state",)
2. GatedDeltaRuleHandler pattern — matches the auto_functionalized_v2
   wrapper that edge decomposition inserts for mutating ops

After edge decomposition the graph looks like:
    auto_func = auto_functionalized_v2(mlx.gated_delta_rule, q=..., ...)
    getitem   = auto_func[0]   # output tensor  (USER_OUTPUT)
    getitem_1 = auto_func[1]   # mutated state  (BUFFER_MUTATION)
    return (getitem_1, getitem)

The pattern handler uses HEAD = getitem[1] (same as ETKVCacheUpdateHandler)
because the partitioner needs the BUFFER_MUTATION node as a proper subgraph
output. getitem[0] is left for the normal _getitem_handler to process.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from torch.fx.node import Node


@torch.library.custom_op("mlx::gated_delta_rule", mutates_args=("state",))
def gated_delta_rule(
    q: Tensor,  # [B, T, Hk, Dk]
    k: Tensor,  # [B, T, Hk, Dk]
    v: Tensor,  # [B, T, Hv, Dv]
    g: Tensor,  # [B, T, Hv] — decay gate
    beta: Tensor,  # [B, T, Hv] — update gate
    state: Tensor,  # [B, Hv, Dv, Dk] — recurrent state (MUTATED in place)
    use_custom_kernel: bool = True,
) -> Tensor:
    """
    Gated delta rule recurrence — sequential scan over T.

    Returns:
        output: [B, T, Hv, Dv]
    """
    B, T_len, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    s = state.clone()

    ys = []
    for t in range(T_len):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = g[:, t]
        beta_t = beta[:, t]

        s = s * g_t[:, :, None, None]
        kv_mem = (s * k_t[:, :, None, :]).sum(dim=-1)
        delta = (v_t - kv_mem) * beta_t[:, :, None]
        s = s + k_t[:, :, None, :] * delta[:, :, :, None]
        y_t = (s * q_t[:, :, None, :]).sum(dim=-1)
        ys.append(y_t)

    state.copy_(s)

    return torch.stack(ys, dim=1)


@torch.library.register_fake("mlx::gated_delta_rule")
def gated_delta_rule_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
    use_custom_kernel: bool = True,
) -> Tensor:
    B, T = q.shape[:2]
    Hv, Dv = v.shape[-2:]
    return v.new_empty(B, T, Hv, Dv)


from executorch.backends.mlx.builder.op_helpers import torch_dtype_to_scalar_type
from executorch.backends.mlx.builder.op_registry import PatternHandler, REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AddNode,
    ExpandDimsNode,
    IdCopyNode,
    IntOrVid,
    MetalKernelNode,
    MultiplyNode,
    ScanNode,
    SubtractNode,
    SumNode,
)
from torch.export.exported_program import ExportedProgram


class GatedDeltaRuleHandler(PatternHandler):
    """
    Pattern for gated delta rule state mutation.

    HEAD = getitem[1] (BUFFER_MUTATION — mutated state)
    BODY = [auto_func_node, getitem_0]

    Both getitem nodes are handled by this pattern to prevent
    _getitem_handler from calling slot_map on auto_func_node
    (which would create a slot on the deferred body node and
    fail _verify_build). The HEAD handler sets slots for both.
    """

    def __init__(
        self,
        head: Node,
        body: List[Node],
        auto_func_node: Node,
        getitem_0: Node,
        q: Node,
        k: Node,
        v: Node,
        g: Node,
        beta: Node,
        state: Node,
    ):
        super().__init__(head, body)
        self.auto_func_node = auto_func_node
        self.getitem_0 = getitem_0
        self.q_node = q
        self.k_node = k
        self.v_node = v
        self.g_node = g
        self.beta_node = beta
        self.state_node = state

    @staticmethod
    def _is_auto_func_gated_delta_rule(node: Node) -> bool:
        if node.op != "call_function":
            return False
        if "auto_functionalized" not in str(node.target):
            return False
        if len(node.args) < 1:
            return False
        func_str = str(node.args[0]) if node.args[0] else ""
        return "gated_delta_rule" in func_str and "mlx" in func_str

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["GatedDeltaRuleHandler"]:
        """
        Match HEAD = getitem[1] from auto_functionalized_v2(gated_delta_rule).
        """
        if head.op != "call_function" or "getitem" not in str(head.target):
            return None
        if len(head.args) < 2 or head.args[1] != 1:
            return None
        if not isinstance(head.args[0], Node):
            return None

        auto_func_node = head.args[0]
        if not cls._is_auto_func_gated_delta_rule(auto_func_node):
            return None

        kwargs = auto_func_node.kwargs
        q = kwargs.get("q")
        k = kwargs.get("k")
        v = kwargs.get("v")
        g = kwargs.get("g")
        beta = kwargs.get("beta")
        all_bases = kwargs.get("_all_bases", [])

        if not all([q, k, v, g, beta]) or not all_bases:
            return None

        state = all_bases[0]

        # Find getitem[0] (output tensor) among auto_func's users
        getitem_0 = None
        for user in auto_func_node.users:
            if (
                user.op == "call_function"
                and "getitem" in str(user.target)
                and len(user.args) >= 2
                and user.args[1] == 0
            ):
                getitem_0 = user
                break

        if getitem_0 is None:
            return None

        return cls(
            head=head,
            body=[auto_func_node, getitem_0],
            auto_func_node=auto_func_node,
            getitem_0=getitem_0,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state=state,
        )

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head

        q_meta = self.q_node.meta["val"]
        Dk = int(q_meta.shape[-1])

        # Read use_custom_kernel from the op's kwargs in the graph
        use_custom_kernel = self.auto_func_node.kwargs.get("use_custom_kernel", True)

        if use_custom_kernel:
            if Dk % 32 != 0:
                raise ValueError(
                    f"MetalKernelNode requires Dk to be a multiple of 32, got Dk={Dk}. "
                    f"Set use_custom_kernel=False to use the ScanNode fallback."
                )
            return self._emit_metal_kernel(P, n)
        return self._emit_scan(P, n)

    def _emit_metal_kernel(self, P: MLXProgramBuilder, n: Node) -> Slot:
        """Emit a fused MetalKernelNode for the gated delta recurrence."""
        from executorch.backends.mlx.serialization.mlx_graph_schema import (
            FullNode,
            MultiplyIntNode,
            SymSizeNode,
        )

        q_slot, k_slot, v_slot, g_slot, beta_slot, state_slot = P.slot_map(
            [
                self.q_node,
                self.k_node,
                self.v_node,
                self.g_node,
                self.beta_node,
                self.state_node,
            ]
        )

        # Extract shapes from metadata
        q_meta = self.q_node.meta["val"]
        v_meta = self.v_node.meta["val"]
        _, _, Hk, Dk = q_meta.shape
        Hv, Dv = v_meta.shape[-2:]
        dtype_int = torch_dtype_to_scalar_type(q_meta.dtype)

        # B and T are potentially dynamic — extract as runtime Vids via SymSizeNode
        _, b_val = P.make_tmp_value_slot()
        P.emit(
            SymSizeNode(
                a=P.slot_to_tid(q_slot),
                dim=0,
                out=P.slot_to_vid(b_val),
            )
        )
        _, t_val = P.make_tmp_value_slot()
        P.emit(
            SymSizeNode(
                a=P.slot_to_tid(q_slot),
                dim=1,
                out=P.slot_to_vid(t_val),
            )
        )

        # grid[2] = B * Hv (computed at runtime)
        _, b_times_hv = P.make_tmp_value_slot()
        P.emit(
            MultiplyIntNode(
                a=P.to_int_or_vid(b_val),
                b=IntOrVid.from_literal(int(Hv)),
                out=P.slot_to_vid(b_times_hv),
            )
        )

        # T as a 0-D int32 tensor for the kernel input (created at runtime from Vid)
        _, t_tensor = P.make_tmp_slot()
        P.emit(
            FullNode(
                out=P.slot_to_tid(t_tensor),
                shape=[],
                v=P.to_float_or_vid(t_val),
                scalar_type=torch_dtype_to_scalar_type(torch.int32),
            )
        )

        # B as IntOrVid for output shapes
        b_iov = P.to_int_or_vid(b_val)
        t_iov = P.to_int_or_vid(t_val)

        # Output slot for y — use existing IO slot if getitem_0 is a graph output,
        # otherwise create a new temp slot.
        out = P.make_or_get_slot(self.getitem_0)

        # Output slot for state_out (carry)
        _, carry = P.make_tmp_slot()

        # Metal kernel source (non-vectorized, no mask variant from mlx-lm)
        source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

        # Output shapes: y=[B,T,Hv,Dv], state_out=[B,Hv,Dv,Dk]
        # B and T are dynamic (Vids), Hv/Dv/Dk are static literals
        output_shapes_flat = [
            # y shape
            b_iov,
            t_iov,
            IntOrVid.from_literal(int(Hv)),
            IntOrVid.from_literal(int(Dv)),
            # state_out shape
            b_iov,
            IntOrVid.from_literal(int(Hv)),
            IntOrVid.from_literal(int(Dv)),
            IntOrVid.from_literal(int(Dk)),
        ]
        output_shape_lengths = [4, 4]

        P.emit(
            MetalKernelNode(
                name="gated_delta_step",
                source=source,
                inputs=[
                    P.slot_to_tid(q_slot),
                    P.slot_to_tid(k_slot),
                    P.slot_to_tid(v_slot),
                    P.slot_to_tid(g_slot),
                    P.slot_to_tid(beta_slot),
                    P.slot_to_tid(state_slot),
                    P.slot_to_tid(t_tensor),
                ],
                outputs=[P.slot_to_tid(out), P.slot_to_tid(carry)],
                grid=[
                    IntOrVid.from_literal(32),
                    IntOrVid.from_literal(int(Dv)),
                    P.to_int_or_vid(b_times_hv),
                ],
                threadgroup=[
                    IntOrVid.from_literal(32),
                    IntOrVid.from_literal(4),
                    IntOrVid.from_literal(1),
                ],
                input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                output_names=["y", "state_out"],
                output_shapes_flat=output_shapes_flat,
                output_shape_lengths=output_shape_lengths,
                output_dtypes=[dtype_int, dtype_int],
                template_arg_names=["InT", "Dk", "Dv", "Hk", "Hv"],
                template_arg_kinds=[2, 0, 0, 0, 0],  # 2=dtype, 0=int
                template_arg_values=[dtype_int, int(Dk), int(Dv), int(Hk), int(Hv)],
            )
        )

        # HEAD is getitem[1] = mutated state → bind to carry
        P.set_slot(n, carry)
        P.set_slot(self.getitem_0, out)

        return carry

    def _emit_scan(self, P: MLXProgramBuilder, n: Node) -> Slot:
        """Emit ScanNode decomposition of the gated delta recurrence."""

        q_slot, k_slot, v_slot, g_slot, beta_slot, state_slot = P.slot_map(
            [
                self.q_node,
                self.k_node,
                self.v_node,
                self.g_node,
                self.beta_node,
                self.state_node,
            ]
        )

        # Carry needs a writable temp slot
        _, carry = P.make_tmp_slot()
        P.emit(IdCopyNode(x=P.slot_to_tid(state_slot), out=P.slot_to_tid(carry)))

        # Sliced temp slots for per-step inputs
        _, q_s = P.make_tmp_slot()
        _, k_s = P.make_tmp_slot()
        _, v_s = P.make_tmp_slot()
        _, g_s = P.make_tmp_slot()
        _, beta_s = P.make_tmp_slot()

        # Output slot for the recurrence output.
        out = P.make_or_get_slot(self.getitem_0)

        # Body temp slots
        _, t0 = P.make_tmp_slot()
        _, t1 = P.make_tmp_slot()
        _, t2 = P.make_tmp_slot()

        with P.new_chain() as body_idx:
            # state = state * g_t[:, :, None, None]
            P.emit(ExpandDimsNode(x=P.slot_to_tid(g_s), out=P.slot_to_tid(t0), axis=-1))
            P.emit(ExpandDimsNode(x=P.slot_to_tid(t0), out=P.slot_to_tid(t0), axis=-1))
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(carry),
                    b=P.slot_to_tid(t0),
                    out=P.slot_to_tid(carry),
                )
            )

            # kv_mem = (state * k_t[:, :, None, :]).sum(-1)
            P.emit(ExpandDimsNode(x=P.slot_to_tid(k_s), out=P.slot_to_tid(t0), axis=-2))
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(carry), b=P.slot_to_tid(t0), out=P.slot_to_tid(t1)
                )
            )
            P.emit(SumNode(x=P.slot_to_tid(t1), out=P.slot_to_tid(t1), axes=[-1]))

            # delta = (v_t - kv_mem) * beta_t[:, :, None]
            P.emit(
                SubtractNode(
                    a=P.slot_to_tid(v_s), b=P.slot_to_tid(t1), out=P.slot_to_tid(t1)
                )
            )
            P.emit(
                ExpandDimsNode(x=P.slot_to_tid(beta_s), out=P.slot_to_tid(t2), axis=-1)
            )
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(t1), b=P.slot_to_tid(t2), out=P.slot_to_tid(t1)
                )
            )

            # state = state + k[:,:,None,:] * delta[:,:,:,None]
            P.emit(ExpandDimsNode(x=P.slot_to_tid(k_s), out=P.slot_to_tid(t2), axis=-2))
            P.emit(ExpandDimsNode(x=P.slot_to_tid(t1), out=P.slot_to_tid(t1), axis=-1))
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(t2), b=P.slot_to_tid(t1), out=P.slot_to_tid(t2)
                )
            )
            P.emit(
                AddNode(
                    a=P.slot_to_tid(carry),
                    b=P.slot_to_tid(t2),
                    out=P.slot_to_tid(carry),
                )
            )

            # y_t = (state * q_t[:,:,None,:]).sum(-1)
            P.emit(ExpandDimsNode(x=P.slot_to_tid(q_s), out=P.slot_to_tid(t0), axis=-2))
            P.emit(
                MultiplyNode(
                    a=P.slot_to_tid(carry), b=P.slot_to_tid(t0), out=P.slot_to_tid(t0)
                )
            )
            P.emit(SumNode(x=P.slot_to_tid(t0), out=P.slot_to_tid(out), axes=[-1]))

        # Emit the ScanNode
        P.emit(
            ScanNode(
                body_chain_idx=body_idx,
                scan_axis=1,
                originals=[
                    P.slot_to_tid(s)
                    for s in [q_slot, k_slot, v_slot, g_slot, beta_slot]
                ],
                sliced=[P.slot_to_tid(s) for s in [q_s, k_s, v_s, g_s, beta_s]],
                outputs=[P.slot_to_tid(out)],
                carry=[P.slot_to_tid(carry)],
            )
        )

        # HEAD is getitem[1] = mutated state → bind to carry
        P.set_slot(n, carry)

        # Set getitem[0] slot → output tensor (for downstream computation)
        P.set_slot(self.getitem_0, out)

        return carry


_registered = False


def register():
    global _registered
    if _registered:
        return
    REGISTRY.register_pattern(name="GATED_DELTA_RULE")(GatedDeltaRuleHandler)
    _registered = True


register()
