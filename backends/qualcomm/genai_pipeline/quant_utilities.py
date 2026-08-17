# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def is_node_src_start_with_name(node: torch.fx.Node, kv_cache_prefix: str) -> bool:
    """
    Return True if any NodeSource in node.meta['from_node'] has a name
    starting with `kv_cache_prefix`. Used to identify K/V cache nodes by their
    "k_" or "v_" name prefix in the traced graph.
    """

    def has_source_name_prefix(
        node_src: torch.fx.traceback.NodeSource, kv_cache_prefix: str
    ) -> bool:

        name = getattr(node_src, "name", None)
        if isinstance(name, str) and name.startswith(kv_cache_prefix):
            return True

        children = getattr(node_src, "from_node", None)
        if not children:
            return False

        for src in children:
            if has_source_name_prefix(src, kv_cache_prefix):
                return True

        return False

    node_srcs = node.meta.get("from_node", None)
    if not node_srcs:
        return False

    return any(
        has_source_name_prefix(node_src, kv_cache_prefix) for node_src in node_srcs
    )


def make_quantizer(
    quant_dtype: Any = None,
    backend: Any = None,
    soc_model: Any = None,
    quant_recipe: Any = None,
    **kwargs: Any,
) -> Any:
    """Create and configure a QNN quantizer."""
    from executorch.backends.qualcomm.export_utils import (
        make_quantizer as _make_quantizer,
    )

    soc_model_str = soc_model.name if hasattr(soc_model, "name") else str(soc_model)
    make_quantizer_kwargs = {
        "backend": backend,
        "soc_model": soc_model_str,
        **kwargs,
    }
    if quant_dtype is not None:
        make_quantizer_kwargs["quant_dtype"] = quant_dtype

    quantizer = _make_quantizer(**make_quantizer_kwargs)
    if quant_recipe is not None:
        recipe_config = (
            quant_recipe.recipe if hasattr(quant_recipe, "recipe") else quant_recipe
        )
        quantizer.set_recipe(recipe_config)
    return quantizer


def save_quantized_module(
    quantized_module: Any,
    example_inputs: Any,
    artifact_dir: str,
) -> Path:
    """Export the calibration graph used for decode SQNR evaluation."""
    from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
        DECODE_QDQ_FILENAME,
    )

    qdq_ep_path = Path(artifact_dir) / DECODE_QDQ_FILENAME
    qdq_ep_path.parent.mkdir(parents=True, exist_ok=True)
    qdq_ep = torch.export.export(quantized_module, example_inputs, strict=True)
    torch.export.save(qdq_ep, qdq_ep_path)
    logger.info("QDQ EP saved to %s", qdq_ep_path)
    return qdq_ep_path


def _logits_io_shape(meta: Dict[str, Any]) -> set:
    """The logits output shape, derived from the graph's metadata."""
    return {
        # logit output
        (
            meta["get_max_batch_size"],
            meta["get_ar_len"],
            meta["get_vocab_size"],
        ),
    }


def _kv_cache_shape(meta: Dict[str, Any]) -> set:
    """The set of valid K/V cache shapes (last two dims), from the metadata.

    A model whose metadata carries ``get_global_head_dim`` has a per-layer
    head_dim (e.g. Gemma 4: sliding=256, full=512), so every head_dim variant
    contributes its own input/output shapes; otherwise a single head_dim is used.
    """
    if "get_global_head_dim" in meta:
        # Gemma 4 has per-layer head_dim: sliding=256, full=512
        kv_head_dims = {
            meta["get_head_dim"],
            meta["get_global_head_dim"],
        }
        kv_cache_shape = set()
        for head_dim in kv_head_dims:
            kv_cache_shape.add((head_dim, meta["get_max_context_len"]))
            kv_cache_shape.add((meta["get_max_context_len"], head_dim))
            kv_cache_shape.add((head_dim, meta["get_ar_len"]))
            kv_cache_shape.add((meta["get_ar_len"], head_dim))
        return kv_cache_shape
    return {
        # single head, kv input
        (meta["get_head_dim"], meta["get_max_context_len"]),
        (meta["get_max_context_len"], meta["get_head_dim"]),
        # single head, kv output
        (meta["get_head_dim"], meta["get_ar_len"]),
        (meta["get_ar_len"], meta["get_head_dim"]),
    }


def save_logits_quant_attrs(
    graph_module: torch.fx.GraphModule, meta: Dict[str, Any]
) -> None:
    """Record the quantized logits scale/zero-point into ``meta`` (in place).

    Scans the graph output for the ``dequantize_per_tensor`` node whose source
    tensor matches the logits shape, and writes ``get_logits_scale`` /
    ``get_logits_zero_point``.
    """
    io_shape = _logits_io_shape(meta)
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for output_node in node.args[0]:
                if (
                    output_node.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ):
                    source_node = output_node.args[0].args[0]
                    if source_node.meta["val"].size() in io_shape:
                        meta["get_logits_scale"] = output_node.args[1]
                        meta["get_logits_zero_point"] = output_node.args[2]
                        break


def save_output_kv_cache_quant_attrs(
    graph_module: torch.fx.GraphModule, meta: Dict[str, Any]
) -> None:
    """Record each output KV-cache node's quant attributes into ``meta``.

    For every graph output whose last two dims match a KV-cache shape, writes
    ``get_kv_output_{i}_quant_attr`` = ``[scale, zero_point, quant_min,
    quant_max, dtype]`` (used by the attention-sink feature).
    """
    from executorch.backends.qualcomm.builders.utils import is_graph_output

    kv_cache_shape = _kv_cache_shape(meta)
    kv_idx = 0
    for node in graph_module.graph.nodes:
        if not is_graph_output(node):
            continue
        cache_output_node = node.args[0].args[0]
        if cache_output_node.meta["val"].size()[-2:] in kv_cache_shape:
            # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
            meta[f"get_kv_output_{kv_idx}_quant_attr"] = [
                node.args[1],
                node.args[2],
                node.args[3],
                node.args[4],
                str(node.args[5]),
            ]
            kv_idx += 1


def encoding_override(  # noqa: C901
    quantized_model: torch.fx.GraphModule,
    unquantized_model: torch.fx.GraphModule,
    n_cache_layers: Optional[int] = None,
) -> None:
    """Copy calibration encodings to a deployed graph.

    Activation and parameter encodings are always copied. Supplying
    ``n_cache_layers`` additionally copies KV-cache output encodings onto the
    deployed graph's cache inputs; ``None`` leaves KV-cache encodings unchanged.
    """
    from executorch.backends.qualcomm.builders.utils import is_graph_output

    pbq_target = {
        torch.ops.torchao.dequantize_affine,
        torch.ops.torchao.quantize_affine,
    }
    pcq_target = {
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    }
    ptq_target = {
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
    }
    qdq_target = pbq_target | pcq_target | ptq_target

    def compare_nodes(quantized_node, unquantized_node):
        def info(node):
            return node.name + (
                str(node.meta["nn_module_stack"].values())
                if node.op == "call_function"
                else ""
            )

        assert info(quantized_node) == info(
            unquantized_node
        ), f"found unmatched order for ops: {quantized_node} vs {unquantized_node}"

    def resolve_param_target(node):
        return (
            node
            if node.op == "call_function" and node.target not in qdq_target
            else resolve_param_target(list(node.users)[0])
        )

    def activation_override(quantized_node, unquantized_node):
        for quantized_user, unquantized_user in zip(
            list(quantized_node.users), list(unquantized_node.users)
        ):
            if "output" == quantized_user.name:
                continue
            assert quantized_user.target == unquantized_user.target, (
                "found unmatched targets: "
                f"{quantized_user.target} vs {unquantized_user.target}"
            )
            if quantized_user.target in qdq_target:
                unquantized_user.args = (
                    unquantized_user.args[0],
                    *quantized_user.args[1:],
                )
                activation_override(quantized_user, unquantized_user)

    def parameter_override(quantized_node, unquantized_node):
        # Some parameters need to be iterated over to retrieve attributes such as static_llama.tok_embedding.weight
        def _get_attr(graph_module: torch.fx.GraphModule, target: str) -> Any:
            attr: Any = graph_module
            for target_atom in target.split("."):
                attr = getattr(attr, target_atom)
            return attr

        def _set_attr(
            graph_module: torch.fx.GraphModule, target: str, replacement: Any
        ) -> Any:
            attr: Any = graph_module
            target_list = target.split(".")
            for target_atom in target_list[:-1]:
                attr = getattr(attr, target_atom)
            setattr(attr, target_list[-1], replacement)

        _set_attr(
            unquantized_model,
            unquantized_node.target,
            _get_attr(quantized_model, quantized_node.target),
        )
        # scale / zero point are part of op's attributes
        if list(quantized_node.users)[0].target in ptq_target:
            activation_override(quantized_node, unquantized_node)

    # copy encoding for hybrid mode
    parameters = [
        {n: resolve_param_target(n) for n in model.graph.nodes if n.op == "get_attr"}
        for model in (quantized_model, unquantized_model)
    ]
    activations = [
        [
            n
            for n in model.graph.nodes
            if n.target not in qdq_target and n.op in {"call_function", "placeholder"}
        ]
        for model in (quantized_model, unquantized_model)
    ]
    # check topology order by node name & nn_module_stack
    for act_quantized, act_unquantized in zip(*activations):
        compare_nodes(act_quantized, act_unquantized)

    for op_quantized, op_unquantized in zip(*[p.values() for p in parameters]):
        compare_nodes(op_quantized, op_unquantized)
    # perform encoding override
    for act_quantized, act_unquantized in zip(*activations):
        activation_override(act_quantized, act_unquantized)

    for param_quantized, param_unquantized in zip(*[p.keys() for p in parameters]):
        parameter_override(param_quantized, param_unquantized)

    if n_cache_layers is not None:
        k_input_cache_nodes = []
        v_input_cache_nodes = []
        for node in unquantized_model.graph.nodes:
            if node.op != "placeholder":
                continue

            if "args_" in node.name:
                args_idx = int(node.name.split("_")[-1])

                if args_idx >= n_cache_layers:
                    v_input_cache_nodes.append(node)
                else:
                    k_input_cache_nodes.append(node)

        if not k_input_cache_nodes or not v_input_cache_nodes:
            raise RuntimeError(
                "KV cache input detection failed. This likely means the model naming "
                "does not match expected prefixes."
            )

        k_output_cache_nodes = []
        v_output_cache_nodes = []
        for node in quantized_model.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if is_node_src_start_with_name(cache_output_node, kv_cache_prefix="k_"):
                k_output_cache_nodes.append(cache_output_node)
            elif is_node_src_start_with_name(cache_output_node, kv_cache_prefix="v_"):
                v_output_cache_nodes.append(cache_output_node)

        if not k_output_cache_nodes or not v_output_cache_nodes:
            raise RuntimeError(
                "KV cache detection failed. This likely means the model naming "
                "does not match expected prefixes."
            )

        for input_k_cache_node, output_k_cache_node in zip(
            k_input_cache_nodes, k_output_cache_nodes
        ):
            activation_override(output_k_cache_node, input_k_cache_node)
        for input_v_cache_node, output_v_cache_node in zip(
            v_input_cache_nodes, v_output_cache_nodes
        ):
            activation_override(output_v_cache_node, input_v_cache_node)

    unquantized_model.recompile()
