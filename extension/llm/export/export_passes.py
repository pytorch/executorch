# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.exir.pass_base import ExportPass
from torch._subclasses import FakeTensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.infra.pass_base import PassResult


def _normalize_dims(tensor: FakeTensor, dim_0: int, dim_1: int):
    """
    Normalize the dimensions of a tensor.
    """
    assert tensor is not None, "Tensor is None"
    ndim = tensor.ndim
    if dim_0 < 0:
        dim_0 = ndim + dim_0
    if dim_1 < 0:
        dim_1 = ndim + dim_1
    assert dim_0 < ndim and dim_1 < ndim, f"Invalid dimensions: {dim_0}, {dim_1}"
    return dim_0, dim_1


class RemoveRedundantTransposes(ExportPass):
    """
    This pass removes redundant transpose nodes in the graph.
    It checks if the next node is also a transpose node and if the two transpose nodes undo each other.
    For example, if the graph has the following nodes:

    node1 = torch.ops.aten.transpose.int(x, 0, 1)
    node2 = torch.ops.aten.transpose.int(node1, 0, 1)

    Then node2's use can be replaced by x

    It will also check for permute nodes
    node1 = torch.ops.aten.permute(x, [0, 2, 1])
    node2 = torch.ops.aten.permute(node1, [0, 2, 1])

    Then also node2's use can be replaced by x

    NB: Does not work for inplace ops or functionalized _copy suffix ops
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph_changed = False
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.transpose.int
            ):
                # Check if the next node is also a transpose node
                tranpose_users = list(node.users.keys())
                dim_0 = node.args[1]
                dim_1 = node.args[2]
                dim_0, dim_1 = _normalize_dims(node.args[0].meta["val"], dim_0, dim_1)

                for user in tranpose_users:
                    if (
                        user.op == "call_function"
                        and user.target == torch.ops.aten.transpose.int
                    ):
                        # Get the arguments of the current and next transpose nodes
                        user_dim_0 = user.args[1]
                        user_dim_1 = user.args[2]
                        user_dim_0, user_dim_1 = _normalize_dims(
                            user.args[0].meta["val"], user_dim_0, user_dim_1
                        )

                        # Check if the two transpose nodes undo each other
                        if dim_0 == user_dim_0 and dim_1 == user_dim_1:
                            graph_changed = True
                            user.replace_all_uses_with(node.args[0])

        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.permute.default
            ):
                # Check if the next node is also a transpose node
                permute_users = list(node.users.keys())
                dim_list = node.args[1]

                for user in permute_users:
                    if (
                        user.op == "call_function"
                        and user.target == torch.ops.aten.permute.default
                    ):
                        # Get the arguments of the current and next transpose nodes
                        user_dim_list = user.args[1]

                        # Check if the two permutes undo each other
                        if dim_list == user_dim_list:
                            graph_changed = True
                            user.replace_all_uses_with(node.args[0])

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, graph_changed)


class ReplaceSDPAWithCustomSDPAPass(ExportPass):
    """
    This pass replaces aten.scaled_dot_product_attention.default with
    llama.custom_sdpa.default.
    If assume_causal_mask is set to True, this pass will ignore any explicit
    masks and simply set is_causal to True in custom_sdpa.
    """

    def __init__(self, assume_causal_mask=False):
        super().__init__()
        self.assume_causal_mask = assume_causal_mask

    def call_operator(self, op, args, kwargs, meta):
        if op != torch.ops.aten.scaled_dot_product_attention.default:
            return super().call_operator(op, args, kwargs, meta)

        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        q, k, v, mask, dropout, is_causal, scale, enable_gqa = self._extract_args(
            args, kwargs
        )

        if self.assume_causal_mask:
            mask = None
            is_causal = True

        if not self._can_rewrite_sdpa(
            q,
            k,
            v,
            mask,
            dropout,
            is_causal,
            enable_gqa,
            assume_causal_mask=self.assume_causal_mask,
        ):
            return super().call_operator(op, args, kwargs, meta)

        qT = self._transpose(q, meta)
        kT = self._transpose(k, meta)
        vT = self._transpose(v, meta)

        if mask is not None:
            mask_fake_tensor = mask.node.meta["val"]
            if mask_fake_tensor.dim() > 2:
                if all(d == 1 for d in mask_fake_tensor.size()[:-2]):
                    mask = super().call_operator(
                        torch.ops.aten.squeeze.dims,
                        (mask, tuple(i for i in range(mask_fake_tensor.dim() - 2))),
                        {},
                        meta,
                    )
                else:
                    logging.info(
                        "ReplaceSDPAWithCustomSDPAPass only supports 2D attention mask."
                    )
                    return super().call_operator(op, args, kwargs, meta)

            # TODO(kimishpatel): Remove once custom SDPA supports boolean mask.
            if mask_fake_tensor.dtype == torch.bool:
                mask = super().call_operator(
                    torch.ops.aten.where.Scalar,
                    (mask, 0.0, float("-inf")),
                    {},
                    meta,
                )

            mask = super().call_operator(
                torch.ops.aten.contiguous.default,
                (mask,),
                {},
                meta,
            )

        custom_sdpa = super().call_operator(
            torch.ops.llama.custom_sdpa.default,
            (qT, kT, vT, 0, mask, dropout, is_causal, scale),
            {},
            meta,
        )
        return self._transpose(custom_sdpa, meta)

    def _can_rewrite_sdpa(
        self,
        q,
        k,
        v,
        mask,
        dropout,
        is_causal,
        enable_gqa,
        *,
        assume_causal_mask,
    ):
        if not (
            q.node.meta["val"].dim()
            == k.node.meta["val"].dim()
            == v.node.meta["val"].dim()
            == 4
        ):
            logging.info("ReplaceSDPAWithCustomSDPAPass only supports 4D QKV inputs.")
            return False

        if not self._supports_head_layout(q, k, v, enable_gqa):
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass requires matching heads or "
                "enable_gqa=True with compatible query/KV heads."
            )
            return False
        if not self._supports_native_shape_contract(q, k, v):
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass requires matching batch shapes "
                "and Q/K/V head dimensions."
            )
            return False
        if dropout != 0.0:
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass does not rewrite SDPA with "
                "non-zero dropout."
            )
            return False
        from executorch.extension.llm.custom_ops.custom_ops import (
            _CUSTOM_SDPA_SUPPORTED_DTYPES,
        )

        if q.node.meta["val"].dtype not in _CUSTOM_SDPA_SUPPORTED_DTYPES:
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass only rewrites float32, "
                "bfloat16, or float16 SDPA inputs."
            )
            return False
        if assume_causal_mask and not self._supports_assume_causal_mask_contract(
            q, k, v
        ):
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass keeps aten SDPA when "
                "assume_causal_mask=True but q_len != kv_len is possible."
            )
            return False
        if not self._supports_unmasked_attention_contract(q, k, v, mask, is_causal):
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass does not rewrite unmasked SDPA "
                "when the native custom-op KV-prefix contract would change "
                "aten semantics."
            )
            return False
        if not self._supports_custom_mask_contract(q, k, mask):
            logging.info(
                "ReplaceSDPAWithCustomSDPAPass only rewrites attention masks "
                "that already match the custom SDPA contract."
            )
            return False
        return True

    def _extract_args(self, args, kwargs):
        q, k, v, *rest = args
        mask = None
        dropout = 0.0
        is_causal = False
        scale = None
        enable_gqa = False
        if len(rest) > 0:
            mask = rest[0]
        if len(rest) > 1:
            dropout = rest[1]
        if len(rest) > 2:
            is_causal = rest[2]
        if "scale" in kwargs:
            scale = kwargs["scale"]
        if "enable_gqa" in kwargs:
            enable_gqa = kwargs["enable_gqa"]

        return q, k, v, mask, dropout, is_causal, scale, enable_gqa

    @staticmethod
    def _known_true(expr) -> bool:
        try:
            return statically_known_true(expr)
        except Exception:
            return False

    @staticmethod
    def _fake_val(x):
        return x.node.meta["val"]

    @classmethod
    def _supports_head_layout(cls, q, k, v, enable_gqa):
        q_heads = cls._fake_val(q).shape[1]
        k_heads = cls._fake_val(k).shape[1]
        v_heads = cls._fake_val(v).shape[1]

        if not cls._known_true(k_heads == v_heads):
            return False
        if cls._known_true(q_heads == k_heads):
            return True
        if cls._known_true(k_heads == 1):
            return True
        if not enable_gqa:
            return False
        return cls._known_true((q_heads % k_heads) == 0)

    @classmethod
    def _supports_native_shape_contract(cls, q, k, v):
        q_shape = cls._fake_val(q).shape
        k_shape = cls._fake_val(k).shape
        v_shape = cls._fake_val(v).shape

        batch_dims_match = all(
            cls._known_true(q_dim == k_dim) and cls._known_true(k_dim == v_dim)
            for q_dim, k_dim, v_dim in zip(q_shape[:-3], k_shape[:-3], v_shape[:-3])
        )
        return (
            batch_dims_match
            and cls._known_true(q_shape[-1] == k_shape[-1])
            and cls._known_true(k_shape[-1] == v_shape[-1])
            and cls._known_true(k_shape[-2] == v_shape[-2])
            and cls._known_true(k_shape[-3] == v_shape[-3])
        )

    @classmethod
    def _supports_unmasked_attention_contract(cls, q, k, v, mask, is_causal):
        if mask is not None:
            return True

        q_shape = cls._fake_val(q).shape
        k_shape = cls._fake_val(k).shape
        v_shape = cls._fake_val(v).shape
        q_seq_len = q_shape[2]
        k_seq_len = k_shape[2]
        v_seq_len = v_shape[2]

        if is_causal:
            return cls._known_true(q_seq_len <= k_seq_len) and cls._known_true(
                q_seq_len <= v_seq_len
            )

        return cls._known_true(q_seq_len == k_seq_len) and cls._known_true(
            k_seq_len == v_seq_len
        )

    @classmethod
    def _supports_assume_causal_mask_contract(cls, q, k, v):
        q_shape = cls._fake_val(q).shape
        k_shape = cls._fake_val(k).shape
        v_shape = cls._fake_val(v).shape
        q_seq_len = q_shape[2]
        k_seq_len = k_shape[2]
        v_seq_len = v_shape[2]
        return cls._known_true(q_seq_len == k_seq_len) and cls._known_true(
            k_seq_len == v_seq_len
        )

    @classmethod
    def _supports_custom_mask_contract(cls, q, k, mask):
        if mask is None:
            return True

        mask_fake_tensor = cls._fake_val(mask)
        if mask_fake_tensor.dim() < 2:
            return False
        if mask_fake_tensor.dim() > 2 and any(
            not cls._known_true(d == 1) for d in mask_fake_tensor.size()[:-2]
        ):
            return False

        if mask_fake_tensor.dtype not in (torch.bool, torch.float32):
            return False

        q_seq_len = cls._fake_val(q).shape[2]
        k_seq_len = cls._fake_val(k).shape[2]
        return cls._known_true(mask_fake_tensor.shape[-2] == q_seq_len) and (
            cls._known_true(mask_fake_tensor.shape[-1] == k_seq_len)
        )

    def _transpose(self, x, meta):
        transpose = super().call_operator(
            torch.ops.aten.transpose.int,
            (x, 1, 2),
            {},
            meta,
        )
        contiguous = super().call_operator(
            torch.ops.aten.contiguous.default,
            (transpose,),
            {},
            meta,
        )
        return contiguous
