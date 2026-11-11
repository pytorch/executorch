# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-component pass configuration for a GenAI model.

Lowering takes a ``dep_table`` and a ``passes_job`` per graph. Both are derived
here from the shard count, the graph's ``meta`` and the quantized-IO dtypes, so
they stay consistent with the compile specs that the same stage builds from the
same shard count.

Pass selection and tagging rules live here. Model-specific module paths and
tagging extras live in ``model_layout``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_ATTENTION_SINK_EVICTOR,
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.model_layout import (
    get_model_layout,
    ModelLayout,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import GRAPH_KV_FORWARD

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle

logger = logging.getLogger(__name__)

# Which tagger a component's ``TagQuantIO`` pass uses. The decoder's inspects
# logits, KV-cache and attention-mask shapes; an encoder's only has to keep the
# tensors around the sharding boundary quantized.
TAGGER_DECODER = "decoder"
TAGGER_ENCODER = "encoder"

# ``meta`` keys this module reads.
META_AR_LEN = "get_ar_len"
# Named as the legacy audio encoder exports it (``audio_encoder.py``'s
# ``get_metadata``), so a bundle's ``meta`` needs no translation on the way in.
META_ENCODER_CTX_SIZE = "get_context_size"
META_HEAD_DIM = "get_head_dim"
META_MAX_BATCH_SIZE = "get_max_batch_size"
META_MAX_CONTEXT_LEN = "get_max_context_len"
META_N_LAYERS = "get_n_layers"
META_VOCAB_SIZE = "get_vocab_size"

_DECODER_SHAPE_META = (
    META_AR_LEN,
    META_HEAD_DIM,
    META_MAX_BATCH_SIZE,
    META_MAX_CONTEXT_LEN,
    META_VOCAB_SIZE,
)

# The graph input the embedding fallback leaves as int64, so ``I64toI32`` has to
# leave it alone.
_TOKENS_NODE = "tokens"


@dataclass(frozen=True)
class PassPolicy:
    """How one component's pass configuration differs from the defaults.

    Attributes:
        tagger: Which ``TagQuantIO`` tagger this component uses, one of
            :data:`TAGGER_DECODER` or :data:`TAGGER_ENCODER`. Also selects the
            layer prefixes the sharding pattern matches.
        tag_quant_io_requires_sharding: Whether quantized IO is only tagged when
            the component is sharded. True for encoders, whose tagging exists to
            keep the tensors either side of a split quantized.
        uses_decoder_boundary: Whether the tagger's shapes and dtypes come from
            the decoder's decode graph. True for the token embedding: legacy
            binds the decoder's ``_tag_ios`` and ``fixed_point_type`` for it.
        skip_tokens_when_embedding_quantized: Whether ``I64toI32`` must leave
            the ``tokens`` input alone when the embedding is quantized.
    """

    tagger: str = TAGGER_ENCODER
    tag_quant_io_requires_sharding: bool = True
    uses_decoder_boundary: bool = False
    skip_tokens_when_embedding_quantized: bool = False


# Mirrors the pass mutations in the legacy ``TextDecoder`` and ``Modality``.
# Components absent from this table use the default policy.
_POLICIES: Dict[str, PassPolicy] = {
    ARTIFACT_TEXT_DECODER: PassPolicy(
        tagger=TAGGER_DECODER,
        tag_quant_io_requires_sharding=False,
        skip_tokens_when_embedding_quantized=True,
    ),
    ARTIFACT_TOK_EMBEDDING: PassPolicy(
        tagger=TAGGER_DECODER,
        tag_quant_io_requires_sharding=False,
        uses_decoder_boundary=True,
        skip_tokens_when_embedding_quantized=True,
    ),
    ARTIFACT_ATTENTION_SINK_EVICTOR: PassPolicy(),
    ARTIFACT_AUDIO_ENCODER: PassPolicy(),
    ARTIFACT_TEXT_ENCODER: PassPolicy(),
    ARTIFACT_VISION_ENCODER: PassPolicy(),
}

_DEFAULT_POLICY = PassPolicy()


def get_pass_policy(artifact_key: str) -> PassPolicy:
    """Look up a component's pass policy.

    Args:
        artifact_key: The component's artifact key (see ``artifact_keys``).

    Returns:
        The component's policy, or the default for an unrecognised key.
    """
    policy = _POLICIES.get(artifact_key)
    if policy is None:
        logger.debug(
            "No pass policy for artifact '%s'; using the defaults", artifact_key
        )
        return _DEFAULT_POLICY

    return policy


def resolve_num_sharding(
    artifact_key: str,
    graphs: Dict[str, "GraphBundle"],
    num_sharding: int,
) -> int:
    """Resolve one component's shard count against its layer count.

    The caller uses the result for both the compile specs and the passes, so
    multiple contexts are requested only when the graph is really split.

    Args:
        artifact_key: The component being built for.
        graphs: The component's graphs, keyed by graph name.
        num_sharding: The requested shard count.

    Returns:
        The requested count, clamped to the layer count. 1 if not sharded.

    Raises:
        ValueError: If sharding is requested and no graph has a layer count.
    """
    if num_sharding <= 1:
        return 1

    layer_counts = {
        bundle.meta[META_N_LAYERS]
        for bundle in graphs.values()
        if META_N_LAYERS in bundle.meta
    }
    if not layer_counts:
        raise ValueError(
            f"sharding '{artifact_key}' into {num_sharding} needs "
            f"'{META_N_LAYERS}' in the graph's metadata"
        )

    n_layers = min(layer_counts)
    if num_sharding > n_layers:
        logger.warning(
            "'%s' has %d layers; clamping the shard count from %d to %d",
            artifact_key,
            n_layers,
            num_sharding,
            n_layers,
        )
        return n_layers

    return num_sharding


def build_pass_config(
    artifact_key: str,
    meta: Dict[str, Any],
    control_args: "ControlArgs",
    num_sharding: int = 1,
    quant_io_dtypes: Optional[Dict[str, Any]] = None,
    shape_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Build one graph's ``(dep_table, passes_job)``.

    Starts from the pass manager's defaults and applies, in the legacy flow's
    order: graph sharding, the embedding fallback's skip node, and quantized-IO
    tagging.

    Args:
        artifact_key: The component this graph belongs to (see
            ``artifact_keys``).
        meta: The graph's metadata constants.
        control_args: The flow's control arguments.
        num_sharding: This component's resolved shard count (see
            :func:`resolve_num_sharding`).
        quant_io_dtypes: The boundary dtypes from the quantization recipe.
            ``None`` means quantization was skipped, so IO stays float32.
        shape_meta: The metadata the tagger's shapes come from, when that is not
            ``meta``. See ``PassPolicy.uses_decoder_boundary``.

    Returns:
        A fresh ``(dep_table, passes_job)`` pair for this graph.
    """
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_qnn_pass_manager_cls,
    )

    # No backend argument, as in the legacy flow: the capture-program passes are
    # the HTP set, and a GPU flow takes the same ones today.
    pass_manager_cls = get_qnn_pass_manager_cls()
    passes_job = pass_manager_cls.get_capture_program_passes()
    dep_table = pass_manager_cls.get_passes_dependency_for_capture_program()

    policy = get_pass_policy(artifact_key)
    layout = get_model_layout(control_args.decoder_model)
    shards = max(1, int(num_sharding))

    if shards > 1:
        _activate_sharding(dep_table, passes_job, policy, layout, meta, shards)

    if policy.skip_tokens_when_embedding_quantized and control_args.embedding_quantize:
        _skip_tokens(passes_job)

    if quant_io_dtypes is None:
        logger.debug(
            "Quantization was skipped for a '%s' graph; leaving its IO float32",
            artifact_key,
        )
    elif policy.tag_quant_io_requires_sharding and shards == 1:
        logger.debug(
            "Component '%s' is not sharded; nothing to tag at a split boundary",
            artifact_key,
        )
    else:
        _activate_quant_io(
            passes_job=passes_job,
            policy=policy,
            layout=layout,
            artifact_key=artifact_key,
            meta=meta if shape_meta is None else shape_meta,
            quant_io_dtypes=quant_io_dtypes,
        )

    return dep_table, passes_job


def build_component_pass_config(
    artifact_key: str,
    graphs: Dict[str, "GraphBundle"],
    control_args: "ControlArgs",
    num_sharding: int = 1,
    decoder_graphs: Optional[Dict[str, "GraphBundle"]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build one component's pass configuration, one entry per graph.

    Each graph gets its own pair: the tagger's shapes depend on the graph's AR
    length, so prefill and decode do not share a ``passes_job``.

    Args:
        artifact_key: The component being built for (see ``artifact_keys``).
        graphs: The component's graphs, keyed by graph name.
        control_args: The flow's control arguments.
        num_sharding: This component's resolved shard count.
        decoder_graphs: The text decoder's graphs. Required for a component
            that uses the decoder's boundary; ignored otherwise.

    Returns:
        A ``(dep_tables, passes_jobs)`` pair, both keyed by graph name.
    """
    decode = _decoder_boundary(artifact_key, decoder_graphs)
    logger.info(
        "Component '%s' of model '%s': %d shard(s), %s",
        artifact_key,
        control_args.decoder_model,
        num_sharding,
        get_model_layout(control_args.decoder_model),
    )

    dep_tables: Dict[str, Any] = {}
    passes_jobs: Dict[str, Any] = {}
    for graph_name, bundle in graphs.items():
        dep_table, passes_job = build_pass_config(
            artifact_key=artifact_key,
            meta=bundle.meta,
            control_args=control_args,
            num_sharding=num_sharding,
            quant_io_dtypes=(
                bundle.quant_io_dtypes if decode is None else decode.quant_io_dtypes
            ),
            shape_meta=None if decode is None else decode.meta,
        )
        dep_tables[graph_name] = dep_table
        passes_jobs[graph_name] = passes_job

    return dep_tables, passes_jobs


def _decoder_boundary(
    artifact_key: str,
    decoder_graphs: Optional[Dict[str, "GraphBundle"]],
) -> Optional["GraphBundle"]:
    """The decode bundle, for a component that tags against the decoder.

    Decode, not prefill, for the same reason ``constant_methods`` come from it:
    decode is the authoritative graph.

    Args:
        artifact_key: The component being built for.
        decoder_graphs: The text decoder's graphs, or ``None``.

    Returns:
        The decode bundle, or ``None`` when this component tags on its own.

    Raises:
        ValueError: If the component needs the decoder's boundary and the
            decode graph is absent. Tagging from its own data would leave the
            boundary float32 without a signal.
    """
    if not get_pass_policy(artifact_key).uses_decoder_boundary:
        return None

    decode = (decoder_graphs or {}).get(GRAPH_KV_FORWARD)
    if decode is None:
        raise ValueError(
            f"'{artifact_key}' is tagged against the decoder's '{GRAPH_KV_FORWARD}' "
            "graph, which is not among the graphs to compile"
        )

    return decode


def _activate_sharding(
    dep_table: Dict[Any, Any],
    passes_job: Dict[Any, Any],
    policy: PassPolicy,
    layout: ModelLayout,
    meta: Dict[str, Any],
    shards: int,
) -> None:
    """Insert ``SplitGraph`` and the two dependency edges it needs.

    ``SplitGraph`` runs after ``FoldQDQ`` so it splits the folded graph, and
    ``TagQuantIO`` after ``SplitGraph`` so the tensors either side of a new
    boundary are tagged.

    Args:
        dep_table: The graph's dependency table, edited in place.
        passes_job: The graph's pass configuration, edited in place.
        policy: The component's policy.
        layout: The model's layout.
        meta: The graph's metadata constants.
        shards: How many shards to split into.

    Raises:
        ValueError: If the layer count is missing from ``meta``.
    """
    from executorch.backends.qualcomm._passes import FoldQDQ, TagQuantIO
    from executorch.extension.llm.custom_ops import model_sharding

    if META_N_LAYERS not in meta:
        raise ValueError(
            f"sharding into {shards} needs '{META_N_LAYERS}' in the graph's "
            "metadata to place the shard boundaries"
        )

    is_decoder = policy.tagger == TAGGER_DECODER
    prefixes = (
        layout.decoder_layer_prefixes if is_decoder else layout.encoder_layer_prefixes
    )
    split_graph, setting = model_sharding.get_split_graph_pass(
        meta[META_N_LAYERS],
        shares=shards,
        pattern=_sharding_pattern(prefixes),
        layer_prefix_offsets=(
            _layer_prefix_offsets(layout, meta) if is_decoder else None
        ),
    )
    passes_job[split_graph] = setting
    dep_table[split_graph] = [FoldQDQ]
    dep_table[TagQuantIO] = [split_graph]


def _sharding_pattern(prefixes: Tuple[str, ...]) -> str:
    """Build the regex matching a layer's index in its module path.

    Args:
        prefixes: The layer module-path prefixes.

    Returns:
        A pattern capturing the layer index.
    """
    prefix_alt = "|".join(re.escape(prefix) for prefix in prefixes)
    return rf"^(?:{prefix_alt})\.(\d+)"


def _layer_prefix_offsets(
    layout: ModelLayout,
    meta: Dict[str, Any],
) -> Optional[Dict[str, int]]:
    """Offset each layer prefix into one shared layer numbering.

    Args:
        layout: The model's layout.
        meta: The graph's metadata constants.

    Returns:
        The offsets, or ``None`` for a decoder whose layers are one sequence.

    Raises:
        ValueError: If an offset's meta key is missing.
    """
    if not layout.layer_offset_meta:
        return None

    offsets = dict.fromkeys(layout.decoder_layer_prefixes, 0)
    for prefix, meta_key in layout.layer_offset_meta:
        if meta_key not in meta:
            raise ValueError(
                f"sharding needs '{meta_key}' in the graph's metadata to offset "
                f"the layers under '{prefix}'"
            )
        offsets[prefix] = meta[meta_key]

    return offsets


def _skip_tokens(passes_job: Dict[Any, Any]) -> None:
    """Leave the ``tokens`` input alone when converting int64 to int32.

    A quantized embedding keeps its lookup in int64, so narrowing that input
    would change what is looked up.

    Args:
        passes_job: The graph's pass configuration, edited in place.
    """
    from executorch.backends.qualcomm._passes import I64toI32
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    )

    passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["skip_node"] = {
        _TOKENS_NODE
    }


def _activate_quant_io(
    passes_job: Dict[Any, Any],
    policy: PassPolicy,
    layout: ModelLayout,
    artifact_key: str,
    meta: Dict[str, Any],
    quant_io_dtypes: Dict[str, Any],
) -> None:
    """Activate ``TagQuantIO`` with this component's tagger.

    Args:
        passes_job: The graph's pass configuration, edited in place.
        policy: The component's policy.
        layout: The model's layout.
        artifact_key: The component being built for.
        meta: The metadata the tagger's shapes come from.
        quant_io_dtypes: The boundary dtypes from the quantization recipe.
    """
    from executorch.backends.qualcomm._passes import TagQuantIO
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_PASS_ACTIVATE_KEY,
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    )

    if policy.tagger == TAGGER_DECODER:
        tagger = _make_decoder_tagger(meta, layout, quant_io_dtypes)
    else:
        tagger = _make_encoder_tagger(artifact_key, meta, quant_io_dtypes)

    passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
    passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
        "get_quant_io_dtype_fn"
    ] = tagger


def _decoder_shapes(meta: Dict[str, Any], layout: ModelLayout) -> Dict[str, Any]:
    """Derive the tensor shapes the decoder's tagger matches on.

    Args:
        meta: The decoder graph's metadata constants.
        layout: The model's layout.

    Returns:
        The ``io_shape``, ``kv_cache_shape``, ``atten_mask_shape`` and
        ``freq_shape`` sets.

    Raises:
        ValueError: If a shape cannot be derived from ``meta``, since tagging
            the wrong tensors is an accuracy loss rather than a failure.
    """
    extra = (layout.extra_head_dim_meta,) if layout.extra_head_dim_meta else ()
    missing = sorted(key for key in _DECODER_SHAPE_META + extra if key not in meta)
    if missing:
        raise ValueError(
            f"cannot tag quantized IO without {missing} in the graph's "
            "metadata; the shapes decide which tensors are quantized at the "
            "graph boundary"
        )

    ar_len = meta[META_AR_LEN]
    max_context_len = meta[META_MAX_CONTEXT_LEN]
    head_dims = {meta[META_HEAD_DIM]} | {meta[key] for key in extra}

    kv_cache_shape = set()
    freq_shape = set()
    for dim in head_dims:
        # single head, kv input and kv output
        kv_cache_shape |= {
            (dim, max_context_len),
            (max_context_len, dim),
            (dim, ar_len),
            (ar_len, dim),
        }
        freq_shape.add((ar_len, dim // 2))

    return {
        # logit output
        "io_shape": {(meta[META_MAX_BATCH_SIZE], ar_len, meta[META_VOCAB_SIZE])},
        "kv_cache_shape": kv_cache_shape,
        # num_heads=1: the mask broadcasts across all heads.
        "atten_mask_shape": {(meta[META_MAX_BATCH_SIZE], 1, ar_len, max_context_len)},
        "freq_shape": freq_shape,
    }


def _tag_decoder_ios(  # noqa: C901 - one branch per legacy tagging rule
    node: Any,
    shapes: Dict[str, Any],
    quant_io_dtypes: Dict[str, Any],
    layout: ModelLayout,
) -> Optional[Any]:
    """Decide whether one decoder node stays quantized at the graph boundary.

    Mirrors the legacy ``TextDecoder._tag_ios`` rule for rule: a tensor that
    should be tagged and is not costs accuracy without failing.

    Args:
        node: The node to classify.
        shapes: The shape sets from :func:`_decoder_shapes`.
        quant_io_dtypes: The boundary dtypes from the quantization recipe.
        layout: The model's layout, for its extra tagging rules.

    Returns:
        The node's quantized IO dtype, or ``None`` to leave it alone.
    """
    from executorch.backends.qualcomm.builders.utils import is_graph_output
    from executorch.exir.dialects._ops import ops as exir_ops

    io_shape = shapes["io_shape"]
    kv_cache_shape = shapes["kv_cache_shape"]
    atten_mask_shape = shapes["atten_mask_shape"]
    freq_shape = shapes["freq_shape"]
    freq_op = {exir_ops.edge.aten.select.int, exir_ops.edge.aten.select_copy.int}

    quant_io_type = None

    if node.op == "placeholder":
        if (
            len(users := list(node.users)) > 0
            and "args_" in node.name
            and users[0].meta["val"].size()[-2:] in kv_cache_shape
        ):
            quant_io_type = quant_io_dtypes["kv_type"]
        elif node.meta["val"].size() in io_shape:
            quant_io_type = quant_io_dtypes["io_type"]
        elif node.meta["val"].size() in atten_mask_shape:
            quant_io_type = quant_io_dtypes["io_type"]

    if is_graph_output(node):
        if node.meta["val"].size()[-2:] in kv_cache_shape:
            quant_io_type = quant_io_dtypes["kv_type"]
        elif node.meta["val"].size() in io_shape:
            quant_io_type = quant_io_dtypes["io_type"]

    # tag sharding io
    if exir_ops.edge.llama.fallback.default in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        quant_io_type = quant_io_dtypes["io_type"]

    # tag select op as quantized tensors for freq_sin and freq_cos. It is caused
    # by sharding
    if node.target in freq_op and node.meta["val"].size() in freq_shape:
        quant_io_type = quant_io_dtypes["io_type"]

    if "stack_trace" in node.meta and any(
        trace in node.meta["stack_trace"] for trace in layout.per_layer_input_traces
    ):
        quant_io_type = quant_io_dtypes["io_type"]

    if any(_is_node_src_start_with_name(node, p) for p in layout.donor_kv_prefixes):
        quant_io_type = quant_io_dtypes["io_type"]

    return quant_io_type


def _make_decoder_tagger(
    meta: Dict[str, Any],
    layout: ModelLayout,
    quant_io_dtypes: Dict[str, Any],
) -> Callable[[Any], Optional[Any]]:
    """Bind the decoder's ``get_quant_io_dtype_fn`` to this graph's shapes.

    Args:
        meta: The decoder graph's metadata constants.
        layout: The model's layout.
        quant_io_dtypes: The boundary dtypes from the quantization recipe.

    Returns:
        A callable taking a node and returning its quantized IO dtype.

    Raises:
        ValueError: If ``quant_io_dtypes`` has no ``kv_type``.
    """
    if "kv_type" not in quant_io_dtypes:
        raise ValueError(
            "the decoder's quantized IO needs a 'kv_type'; the recipe exposed "
            f"only {sorted(quant_io_dtypes)}"
        )

    return partial(
        _tag_decoder_ios,
        shapes=_decoder_shapes(meta, layout),
        quant_io_dtypes=quant_io_dtypes,
        layout=layout,
    )


def _tag_encoder_ios(
    node: Any,
    io_type: Any,
    ctx_size: Optional[int],
) -> Optional[Any]:
    """Decide whether one encoder node stays quantized at a split boundary.

    Args:
        node: The node to classify.
        io_type: The activation dtype from the encoder's recipe.
        ctx_size: GraniteSpeech's encoder context size, or ``None`` to skip the
            attention-distance rule.

    Returns:
        The node's quantized IO dtype, or ``None`` to leave it alone.
    """
    import torch

    from executorch.exir.dialects._ops import ops as exir_ops

    quant_io_type = None

    # tag sharding io
    if exir_ops.edge.llama.fallback.default in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        quant_io_type = io_type

    # tag _to_copy op as quantized tensors for attn dist. It is caused by
    # sharding
    if (
        ctx_size is not None
        and node.target == exir_ops.edge.aten._to_copy.default
        and node.meta["val"].size() == (ctx_size, ctx_size)
    ):
        quant_io_type = torch.int32

    return quant_io_type


def _make_encoder_tagger(
    artifact_key: str,
    meta: Dict[str, Any],
    quant_io_dtypes: Dict[str, Any],
) -> Callable[[Any], Optional[Any]]:
    """Bind an encoder's ``get_quant_io_dtype_fn``.

    Args:
        artifact_key: The component being built for.
        meta: The graph's metadata constants.
        quant_io_dtypes: The boundary dtypes from the quantization recipe.

    Returns:
        A callable taking a node and returning its quantized IO dtype.
    """
    # GraniteSpeech's attention-distance tensor is square in the encoder's
    # context size. Absent from ``meta`` today, which skips the rule.
    ctx_size = None
    if artifact_key == ARTIFACT_AUDIO_ENCODER:
        ctx_size = meta.get(META_ENCODER_CTX_SIZE)
        if ctx_size is None:
            logger.debug(
                "No '%s' in the audio encoder's metadata; not tagging its "
                "attention-distance tensor",
                META_ENCODER_CTX_SIZE,
            )

    return partial(
        _tag_encoder_ios, io_type=quant_io_dtypes["io_type"], ctx_size=ctx_size
    )


def _is_node_src_start_with_name(node: Any, prefix: str) -> bool:
    """Whether any of a node's sources is named starting with ``prefix``.

    Args:
        node: The node to inspect.
        prefix: The source-name prefix to look for.

    Returns:
        ``True`` if a source, at any depth, has a matching name.
    """

    def has_source_name_prefix(node_src: Any) -> bool:
        name = getattr(node_src, "name", None)
        if isinstance(name, str) and name.startswith(prefix):
            return True

        children = getattr(node_src, "from_node", None)
        if not children:
            return False

        return any(has_source_name_prefix(src) for src in children)

    node_srcs = node.meta.get("from_node", None)
    if not node_srcs:
        return False

    return any(has_source_name_prefix(node_src) for node_src in node_srcs)
