# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.vgf.tosa_custom_sdpa import (
    build_custom_sdpa_payload,
    CUSTOM_SDPA_DOMAIN_NAME,
    custom_sdpa_operator_name,
    custom_sdpa_tosa_fake_impl,
    encode_payload,
    ensure_custom_sdpa_registered,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _set_fake_tensor_meta(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if isinstance(value, list):
        if value:
            node.meta["tensor_meta"] = _extract_tensor_metadata(value[0])
    else:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


class RewriteCustomSDPAToTosaCustomPass(ArmPass):
    """Rewrite supported decode-only SDPA nodes to ``tosa.CUSTOM``.

    This pass handles only the narrow VGF shader contract.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _is_custom_sdpa_target(target: object) -> bool:
        return getattr(target, "__name__", None) == "llama.custom_sdpa.default"

    @classmethod
    def _custom_sdpa_target(cls):
        ensure_custom_sdpa_registered()
        try:
            return exir_ops.edge.llama.custom_sdpa.default
        except AttributeError:
            return None

    @staticmethod
    def _get_permute_dims(node: torch.fx.Node) -> list[int] | None:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.permute_copy.default
            or len(node.args) != 2
            or not isinstance(node.args[1], list)
        ):
            return None
        if not all(isinstance(dim, int) for dim in node.args[1]):
            return None
        return list(node.args[1])

    @staticmethod
    def _is_bshd_permute(node: object) -> bool:
        return isinstance(node, torch.fx.Node) and (
            RewriteCustomSDPAToTosaCustomPass._get_permute_dims(node) == [0, 2, 1, 3]
        )

    @classmethod
    def _unwrap_layout_input(cls, node: torch.fx.Node) -> torch.fx.Node | None:
        if cls._is_bshd_permute(node) and isinstance(node.args[0], torch.fx.Node):
            return node.args[0]
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.dim_order_ops._clone_dim_order.default
            or len(node.args) != 1
            or not isinstance(node.args[0], torch.fx.Node)
        ):
            return None
        permute = node.args[0]
        if not cls._is_bshd_permute(permute) or not isinstance(
            permute.args[0], torch.fx.Node
        ):
            return None
        return permute.args[0]

    @classmethod
    def _find_output_restore(
        cls, node: torch.fx.Node
    ) -> tuple[torch.fx.Node, list[torch.fx.Node]] | None:
        users = list(node.users)
        if len(users) != 1 or not cls._is_bshd_permute(users[0]):
            return None
        permute = users[0]
        return permute, [permute]

    @staticmethod
    def _extract_operator_args(
        node: torch.fx.Node,
    ) -> (
        tuple[
            torch.fx.Node,
            torch.fx.Node,
            torch.fx.Node,
            object,
            object,
            object,
            object,
            object,
        ]
        | None
    ):
        if len(node.args) < 7:
            return None

        query, key, value, start_pos, attn_mask, dropout_p, is_causal, *rest = node.args
        scale = rest[0] if rest else None
        if not isinstance(query, torch.fx.Node):
            return None
        if not isinstance(key, torch.fx.Node):
            return None
        if not isinstance(value, torch.fx.Node):
            return None
        return query, key, value, start_pos, attn_mask, dropout_p, is_causal, scale

    @staticmethod
    def _supports_decode_shape(
        query_val: torch.Tensor,
        key_val: torch.Tensor,
        value_val: torch.Tensor,
        *,
        layout: str,
    ) -> bool:
        if not all(
            all(isinstance(dim, int) for dim in tensor.shape)
            for tensor in (query_val, key_val, value_val)
        ):
            return False
        if not (
            query_val.dim() == 4
            and key_val.dim() == 4
            and value_val.dim() == 4
            and tuple(key_val.shape) == tuple(value_val.shape)
            and query_val.shape[0] == key_val.shape[0]
            and key_val.shape[3] == query_val.shape[3]
        ):
            return False
        if layout == "BSHD":
            return (
                query_val.shape[1] == 1 and query_val.shape[2] % key_val.shape[2] == 0
            )
        if layout == "BHSD":
            return (
                query_val.shape[2] == 1 and query_val.shape[1] % key_val.shape[1] == 0
            )
        return False

    @classmethod
    def supports_tosa_custom(cls, node: torch.fx.Node) -> bool:
        """Return whether ``llama.custom_sdpa`` matches the VGF shader contract.

        The current VGF custom SDPA path is intentionally narrow. It accepts:
        decode-style causal attention only, no explicit attention mask, zero
        dropout, 4D BSHD/BHSD tensors, matching K/V shapes, and grouped-query
        compatible head ratios.

        """
        if not cls._is_custom_sdpa_target(node.target):
            return False
        return cls._get_supported_inputs(node) is not None

    @classmethod
    def _get_supported_inputs(cls, node: torch.fx.Node) -> (
        tuple[
            torch.fx.Node,
            torch.fx.Node,
            torch.fx.Node,
            int,
            float,
            bool,
            float | None,
            str,
        ]
        | None
    ):
        extracted = cls._extract_operator_args(node)
        if extracted is None:
            return None

        query, key, value, start_pos, attn_mask, dropout_p, is_causal, scale = extracted
        if attn_mask is not None:
            return None
        if not isinstance(start_pos, int):
            return None
        if not isinstance(dropout_p, (float, int)) or float(dropout_p) != 0.0:
            return None
        if is_causal is not True:
            return None
        if scale is not None and not isinstance(scale, (float, int)):
            return None

        restore_output = cls._find_output_restore(node)
        original_query = cls._unwrap_layout_input(query)
        original_key = cls._unwrap_layout_input(key)
        original_value = cls._unwrap_layout_input(value)

        use_original_layout = restore_output is not None and all(
            isinstance(arg, torch.fx.Node)
            for arg in (original_query, original_key, original_value)
        )
        if use_original_layout:
            assert original_query is not None
            assert original_key is not None
            assert original_value is not None
            query = original_query
            key = original_key
            value = original_value
            layout = "BHSD"
        else:
            layout = "BSHD"

        query_val = query.meta.get("val")
        key_val = key.meta.get("val")
        value_val = value.meta.get("val")
        if not all(
            isinstance(val, torch.Tensor) for val in (query_val, key_val, value_val)
        ):
            return None
        assert isinstance(query_val, torch.Tensor)
        assert isinstance(key_val, torch.Tensor)
        assert isinstance(value_val, torch.Tensor)

        if not cls._supports_decode_shape(
            query_val,
            key_val,
            value_val,
            layout=layout,
        ):
            return None

        return (
            query,
            key,
            value,
            start_pos,
            float(dropout_p),
            is_causal,
            None if scale is None else float(scale),
            layout,
        )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        modified = False
        if not any(self._is_custom_sdpa_target(node.target) for node in graph.nodes):
            return PassResult(graph_module, False)
        target = self._custom_sdpa_target()
        if target is None:
            return PassResult(graph_module, False)
        for node in list(graph.nodes):
            if node.target != target:
                continue
            if not self.allowed_to_transform(node.meta):
                continue

            restore_output = self._find_output_restore(node)
            supported = self._get_supported_inputs(node)
            if supported is None:
                continue

            query, key, value, start_pos, dropout_p, is_causal, scale, layout = (
                supported
            )
            payload = build_custom_sdpa_payload(
                tuple(query.meta["val"].shape),
                tuple(key.meta["val"].shape),
                tuple(value.meta["val"].shape),
                start_pos,
                dropout_p,
                is_causal,
                scale,
                layout=layout,
            )
            implementation_attrs = encode_payload(payload)
            fake_outputs = custom_sdpa_tosa_fake_impl(
                [query.meta["val"], key.meta["val"], value.meta["val"]],
                custom_sdpa_operator_name(),
                CUSTOM_SDPA_DOMAIN_NAME,
                implementation_attrs,
            )

            with graph.inserting_before(node):
                tosa_custom = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([query, key, value],),
                    kwargs={
                        "operator_name": custom_sdpa_operator_name(),
                        "domain_name": CUSTOM_SDPA_DOMAIN_NAME,
                        "implementation_attrs": implementation_attrs,
                    },
                )
                tosa_custom.meta = dict(node.meta)
                _set_fake_tensor_meta(tosa_custom, fake_outputs)

                output = graph.call_function(operator.getitem, args=(tosa_custom, 0))
                output.meta = dict(node.meta)
                _set_fake_tensor_meta(output, fake_outputs[0])

            if layout == "BHSD" and restore_output is not None:
                replacement_target, stale_nodes = restore_output
                output.meta = dict(replacement_target.meta)
                _set_fake_tensor_meta(output, fake_outputs[0])
                replacement_target.replace_all_uses_with(output)
                for stale_node in reversed(stale_nodes):
                    graph.erase_node(stale_node)
            else:
                node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
