#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry-point for the Gemma 4 backbone static-transformer export binary.

Exports the Gemma 4 E2B/E4B backbone (text decoder) as a single .pte lowered
to the CoreML CPU_AND_NE delegate (Apple Neural Engine).

`--checkpoint_path` accepts either a local HuggingFace checkpoint directory
or a Manifold path (`manifold://bucket/path/to/checkpoint`); Manifold inputs
are streamed via `convert_weights._load_safetensors_weights`.

Usage:
    # E2B (default), local checkpoint:
    buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4_static_transformer -- \
        --checkpoint_path /tmp/gemma4-e2b-it \
        --output_path /tmp/gemma4_backbone_coreml.pte

    # E2B, Manifold checkpoint:
    buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4_static_transformer -- \
        --checkpoint_path manifold://bucket/path/to/gemma4-e2b-it \
        --output_path /tmp/gemma4_backbone_coreml.pte

    # E4B:
    buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4_static_transformer -- \
        --checkpoint_path /tmp/gemma4-e4b-it \
        --variant e4b \
        --output_path /tmp/gemma4_e4b_backbone_coreml.pte
"""

import argparse
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import coremltools as ct
import torch
from executorch.examples.models.gemma4.common_utils import (
    resolve_local_path,
    setup_path_manager,
)
from iopath.common.file_io import PathManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathmgr: PathManager = setup_path_manager()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Gemma 4 backbone (text decoder) to CoreML CPU_AND_NE PTE"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to HuggingFace checkpoint directory. "
        "Accepts either a local path or a Manifold path "
        "(manifold://bucket/path/to/checkpoint).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="e2b",
        choices=["e2b", "e4b"],
        help="Model variant (default: e2b)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/gemma4_backbone_coreml.pte",
        help="Output path for .pte file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length for text decoder KV cache",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=770,
        help="Example sequence length used during export (text + audio tokens)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        default=None,
        help="Static-attention KV cache length per layer. Defaults to max_seq_len.",
    )
    parser.add_argument(
        "--generate_full_logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output full per-token logits. When False, emit only the last token "
        "logit and add `last_valid_token_pos` to attn_options.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="8da4w+emb8",
        choices=["8da4w+emb8", "8da4w+emb4", "8da8w+emb8", "none"],
        help="Quantization spec (default: 8da4w+emb8)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for INT4 weight quantization (default: 128)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Model precision; ANE requires fp16 (default: fp16)",
    )
    parser.add_argument(
        "--tied_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie embed_tokens + lm_head weights to reduce model size. "
        "Requires C++ runner with TorchAO shared embedding kernels.",
    )
    parser.add_argument(
        "--quantize_kv_cache",
        action="store_true",
        default=False,
        help="Use INT8 quantized KV cache to reduce memory for long sequences.",
    )
    parser.add_argument(
        "--use_coreml_mlmodelc",
        action="store_true",
        default=False,
        help="Use mlmodelc (precompiled) format for CoreML.",
    )
    parser.add_argument(
        "--skip_embedding_delegation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip delegating embedding ops to CoreML; let them run on CPU.",
    )
    return parser.parse_args()


@dataclass
class ExportConfig:
    """Configuration object holding all state needed for model export."""

    args: argparse.Namespace
    static_transformer: torch.nn.Module
    model_wrapper: Any
    example_inputs: Any
    dynamic_shapes: Any
    timing_info: Dict[str, float] = field(default_factory=dict)


def _determine_constants(args: argparse.Namespace):
    """Token / float dtype + mask sentinel for the static-attention export path."""
    token_dtype = torch.int32
    float_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.precision]
    mask_val = float("-inf")
    return token_dtype, float_dtype, mask_val


def get_static_example_inputs(
    static_transformer: torch.nn.Module,
    input_len: int,
    cache_len: int,
    args: argparse.Namespace,
):
    """Build (tokens, attn_options) example inputs for Gemma 4 export.

    Gemma4TextModel.forward takes `(input_ids, attn_options, inputs_embeds)`.
    `attn_options` is a plain dict; the only key the text model reads is
    `input_pos`. KV-cache shapes are owned by Gemma4KVCache (registered as
    buffers on each Gemma4Attention), so no IOManager wiring is needed.
    """
    token_dtype, _, _ = _determine_constants(args)
    return (
        torch.zeros(1, input_len, dtype=token_dtype),
        {"input_pos": torch.arange(input_len, dtype=torch.long)},
    )


def _load_static_transformer(args: argparse.Namespace):
    """Load Gemma 4 text decoder backbone, applying quantization transforms."""
    from executorch.examples.models.gemma4.quant_utils import (
        apply_embedding_quantization,
        apply_linear_quantization,
        parse_quantize,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
        Gemma4Config,
    )
    from executorch.examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model

    config = Gemma4Config.from_config(args.variant)
    config.use_kv_cache = True
    config.max_seq_len = args.max_seq_len
    config.enable_dynamic_shape = True
    # Use index_copy_ instead of narrow().copy_() for KV-cache writes. The
    # narrow path produces start_pos = input_pos[0].item() (an unbacked SymInt
    # u700) and writes via aten.slice_scatter, which trips both the CoreML
    # partitioner's SymInt filter and the lowered_backend_module
    # _get_new_signature assertion (partition output → slice_scatter user with
    # no intervening getitem). index_copy_ functionalizes to aten.index_put
    # with all-tensor args — no SymInts, no slice_scatter.
    config.use_index_copy_for_kv_cache = True

    torch_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.precision]

    model_wrapper = Gemma4Model(
        config=config,
        checkpoint_path=resolve_local_path(pathmgr, args.checkpoint_path),
        dtype=torch_dtype,
    )
    static_transformer = model_wrapper.get_eager_model()
    static_transformer.eval()

    linear_quant, emb_quant = parse_quantize(args.quantize)
    if emb_quant and linear_quant and args.tied_embedding:
        # Share embed_tokens + lm_head via TiedEmbeddingQuantizer.
        from torchao.prototype.quantization.embedding.api import TiedEmbeddingQuantizer
        from torchao.quantization.granularity import PerAxis

        weight_dtype = torch.int4 if emb_quant == "emb4" else torch.int8
        logger.info(f"Applying tied embedding (embed_tokens + lm_head, {emb_quant})...")
        TiedEmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0),
        ).quantize(
            static_transformer,
            embedding_to_unembedding={
                "model.self_decoder.embed_tokens": "model.lm_head",
            },
        )
        # Quantize embed_tokens_per_layer with llama's EmbeddingQuantHandler.
        static_transformer = apply_embedding_quantization(static_transformer, emb_quant)
        static_transformer.eval()
    else:
        if emb_quant:
            static_transformer = apply_embedding_quantization(
                static_transformer, emb_quant
            )
            static_transformer.eval()
    if args.quantize_kv_cache:
        from executorch.examples.models.gemma4.text_decoder.gemma4_attention import (
            replace_kv_cache_with_quantized_kv_cache,
        )

        logger.info("Replacing KV cache with INT8 quantized KV cache...")
        static_transformer = replace_kv_cache_with_quantized_kv_cache(
            static_transformer
        )
        static_transformer.eval()

    if linear_quant:
        static_transformer = apply_linear_quantization(
            static_transformer, linear_quant, group_size=args.group_size
        )
        static_transformer.eval()

    return static_transformer, model_wrapper


def _build_export_config(args: argparse.Namespace) -> ExportConfig:
    """Build export configuration: load model, prepare example inputs and dynamic shapes."""
    timing_info: Dict[str, float] = {}

    start_time = time.time()
    static_transformer, model_wrapper = _load_static_transformer(args)
    timing_info["Model Loading"] = time.time() - start_time
    logger.info(f"Model loading took {timing_info['Model Loading']:.4f} seconds.")

    cache_len = args.cache_len if args.cache_len is not None else args.max_seq_len
    example_inputs = get_static_example_inputs(
        static_transformer, args.seq_len, cache_len, args
    )
    # Static attention exports use fixed input/cache shapes — no dynamic dims.
    dynamic_shapes = None

    return ExportConfig(
        args=args,
        static_transformer=static_transformer,
        model_wrapper=model_wrapper,
        example_inputs=example_inputs,
        dynamic_shapes=dynamic_shapes,
        timing_info=timing_info,
    )


def _export_model(export_config: ExportConfig):
    """Export the static transformer to an ExportedProgram."""
    args = export_config.args
    static_transformer = export_config.static_transformer
    example_inputs = export_config.example_inputs
    dynamic_shapes = export_config.dynamic_shapes

    tokens, attn_options = example_inputs
    start_time = time.time()
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            ep = torch.export.export(
                static_transformer,
                (tokens,),
                kwargs={"attn_options": attn_options},
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
    export_config.timing_info["Export"] = time.time() - start_time
    logger.info(
        f"torch.export.export took {export_config.timing_info['Export']:.4f} seconds."
    )

    logger.info("Text decoder exported")
    # Free the eager model now that we have the ExportedProgram.
    export_config.static_transformer = None
    export_config.model_wrapper = None
    del static_transformer
    gc.collect()

    return ep


def _patch_coremltools_inplace_assertion() -> None:
    """Make coremltools' is_torch_fx_node_supported tolerant of in-place ops.

    Upstream coremltools asserts when it sees a node whose target name ends with
    `_` (in-place ops). The KV cache update path in `Gemma4KVCache` produces such
    nodes (`copy_`, `index_copy_`); they remain in the graph because we set
    `take_over_mutable_buffer=False`. With the assertion in place, the partitioner
    crashes before it can route those nodes to CPU. Returning False lets the
    partitioner run them outside the CoreML delegate.

    The CoreML partitioner calls
    `ct.converters.mil.frontend.torch.is_torch_fx_node_supported`, which is
    re-exported from `torch_op_registry`. We patch BOTH attributes so the patched
    version is the one the partitioner sees.
    """
    from coremltools.converters.mil.frontend import torch as ct_torch  # pyre-ignore[21]
    from coremltools.converters.mil.frontend.torch import (  # pyre-ignore[21]
        torch_op_registry,
    )

    if getattr(torch_op_registry, "_inplace_assertion_patched", False):
        return

    original = torch_op_registry.is_torch_fx_node_supported

    def safe_is_torch_fx_node_supported(node):
        try:
            return original(node)
        except AssertionError:
            return False

    torch_op_registry.is_torch_fx_node_supported = safe_is_torch_fx_node_supported
    ct_torch.is_torch_fx_node_supported = safe_is_torch_fx_node_supported
    torch_op_registry._inplace_assertion_patched = True


def _patch_coremltools_index_put_whole_slice() -> None:
    """Replace CoreML's `index_put` op handler to support multi-position writes
    with leading-None indices (the `index_copy_(dim, input_pos, src)` pattern).

    With `use_index_copy_for_kv_cache=True`, the cache update emits
    `aten.index_put.default(self, [None, None, input_pos, None], k_val,
    accumulate=False)` where `input_pos = arange(seq_len)` is a rank-1 tensor
    of length seq_len > 1 and the leading two indices are None (whole slice).

    coremltools' upstream handler can't handle this:
      - Its `_try_whole_slice` path squeezes the index and feeds it into a
        rank-0 `mb.concat`, which fails when the squeeze is a no-op:
          `ValueError: Input squeeze_0 has rank 1 != other inputs rank 0`
      - Its scatter_nd fallback assumes `indices[0]` is non-None and derefs
        `indices[0].sym_type.get_primitive()`, which crashes on None.

    We re-register `index_put` with `override=True`. For the
    "leading Nones + one rank-1 index + trailing None" pattern we permute
    the indexed dim to dim 0, do a single `mb.scatter_nd` with the rank-1
    index expanded to shape (N, 1), and permute back. For the fully-specified
    int-index pattern we fall back to the standard scatter_nd shape. Bool
    indices and other shapes raise NotImplementedError so we don't silently
    miscompile.
    """
    from coremltools.converters.mil.frontend.torch import ops as ct_ops  # pyre-ignore[21]
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (  # pyre-ignore[21]
        register_torch_op,
    )

    if getattr(ct_ops, "_index_put_whole_slice_patched", False):
        return

    @register_torch_op(override=True)
    def index_put(context, node):
        from coremltools.converters.mil.frontend.torch.ops import (  # pyre-ignore[21]
            _get_inputs,
            _get_kwinputs,
        )
        from coremltools.converters.mil.mil import Builder as mb  # pyre-ignore[21]
        from coremltools.converters.mil.mil import types  # pyre-ignore[21]

        inputs = _get_inputs(context, node, expected=(3, 4))
        x = inputs[0]
        indices = inputs[1]
        values = inputs[2]
        accumulate_var = inputs[3] if len(inputs) > 3 else False
        accumulate = _get_kwinputs(
            context, node, "accumulate", default=[accumulate_var]
        )[0]
        if hasattr(accumulate, "val"):
            accumulate = accumulate.val
        if accumulate:
            raise NotImplementedError(
                "index_put with accumulate=True is not supported by this patch"
            )

        # Identify the single non-None index and its dim.
        non_none_dims = [i for i, idx in enumerate(indices) if idx is not None]
        if len(non_none_dims) != 1:
            raise NotImplementedError(
                f"index_put patch supports exactly one non-None index, got "
                f"{len(non_none_dims)} (indices structure: "
                f"{['None' if i is None else 'Var' for i in indices]})"
            )
        scatter_dim = non_none_dims[0]
        scatter_index = indices[scatter_dim]

        if types.is_bool(scatter_index.sym_type.get_primitive()):
            raise NotImplementedError(
                "index_put patch does not support bool indices"
            )

        # Bring scatter_dim to dim 0 of both x and values, do the scatter, then
        # permute back. scatter_nd expects index shape (N, K) where K is the
        # number of leading dims being indexed; we want K=1 (one indexed dim).
        x_rank = x.rank
        if scatter_dim != 0:
            perm = [scatter_dim] + [d for d in range(x_rank) if d != scatter_dim]
            inv_perm = [perm.index(d) for d in range(x_rank)]
            x_perm = mb.transpose(x=x, perm=perm)
            values_perm = mb.transpose(x=values, perm=perm)
        else:
            perm = list(range(x_rank))
            inv_perm = perm
            x_perm = x
            values_perm = values

        # scatter_nd index: (N,) -> (N, 1)
        idx_2d = mb.expand_dims(x=scatter_index, axes=[-1])

        scattered = mb.scatter_nd(
            data=x_perm,
            indices=idx_2d,
            updates=values_perm,
            mode="update",
        )

        if scatter_dim != 0:
            result = mb.transpose(x=scattered, perm=inv_perm, name=node.name)
        else:
            result = mb.identity(x=scattered, name=node.name)
        context.add(result)

    ct_ops._index_put_whole_slice_patched = True


def _patch_coremltools_int8_dtype() -> None:
    """Add torch.int8 / torch.uint8 to coremltools' TORCH_DTYPE_TO_MIL_DTYPE map.

    8da4w / emb8 quantization produces int8 weights that get lifted as inputs of
    CoreML-delegated subgraphs. coremltools' default `TORCH_DTYPE_TO_MIL_DTYPE`
    only covers fp16/fp32/int16/int32 (+ int64/float64 fallbacks); torch.int8
    raises `KeyError: torch.int8` in `_construct_ct_tensor_type_from_torch`.

    MIL has native `types.int8` / `types.uint8`, so the mapping is well-defined —
    upstream just never registered it for the EXIR ingest path.
    """
    from coremltools.converters.mil.frontend.torch import utils as ct_utils  # pyre-ignore[21]
    from coremltools.converters.mil.mil import types  # pyre-ignore[21]

    if getattr(ct_utils, "_int8_dtype_patched", False):
        return

    ct_utils.TORCH_DTYPE_TO_MIL_DTYPE.setdefault(torch.int8, types.int8)
    ct_utils.TORCH_DTYPE_TO_MIL_DTYPE.setdefault(torch.uint8, types.uint8)
    ct_utils._int8_dtype_patched = True


def _patch_coreml_partitioner_skip_symint_args() -> None:
    """Force CoreMLPartitioner to skip nodes whose args carry SymInt/SymBool/SymFloat.

    The KV-cache update / index path in Gemma4Attention produces unbacked
    SymInts (e.g. `u700`) that flow through ops the partitioner happily marks
    as CoreML-supported. After the partition splits the graph, those SymInts
    become placeholder values of the delegated subgraph, and coremltools'
    `_extract_inputs_from_exir_program` raises:

      NotImplementedError: Placeholder val must be a tensor or fake tensor,
      but got type torch.SymInt, value u700

    Upstream `_OperatorsSupportedForCoreMLBackend.should_override_support`
    has the right filter commented out (gated on `lower_full_graph=False`,
    which is our case) with TODO 'enable this after bugs in ExecuTorch's
    partitioner are fixed'. We wrap the method to add the same check.
    """
    from executorch.backends.apple.coreml.partition.coreml_partitioner import (  # pyre-ignore[21]
        _OperatorsSupportedForCoreMLBackend,
    )

    if getattr(_OperatorsSupportedForCoreMLBackend, "_symint_filter_patched", False):
        return

    original = _OperatorsSupportedForCoreMLBackend.should_override_support

    def patched_should_override_support(self, node) -> bool:
        if original(self, node):
            return True
        if not self.lower_full_graph and any(
            isinstance(arg, torch.fx.Node)
            and isinstance(
                arg.meta.get("val", None),
                (torch.SymInt, torch.SymBool, torch.SymFloat),
            )
            for arg in node.args
        ):
            self.log_once(
                "Skipping op for CoreML delegation because it contains symbolic args: "
                + getattr(node.target, "__name__", str(node.target))
            )
            return True
        return False

    _OperatorsSupportedForCoreMLBackend.should_override_support = (
        patched_should_override_support
    )
    _OperatorsSupportedForCoreMLBackend._symint_filter_patched = True


def _build_coreml_partitioner(args: argparse.Namespace):
    """Build a CoreML partitioner targeting the CPU_AND_NE compute unit."""
    from executorch.backends.apple.coreml.compiler.coreml_preprocess import (  # pyre-ignore[21]
        CoreMLBackend,
    )
    from executorch.backends.apple.coreml.partition import (  # pyre-ignore[21]
        CoreMLPartitioner,
    )

    _patch_coremltools_inplace_assertion()
    _patch_coremltools_int8_dtype()
    _patch_coreml_partitioner_skip_symint_args()
    _patch_coremltools_index_put_whole_slice()

    coreml_model_type = (
        CoreMLBackend.MODEL_TYPE.COMPILED_MODEL
        if args.use_coreml_mlmodelc
        else CoreMLBackend.MODEL_TYPE.MODEL
    )

    # ANE only supports FLOAT16 natively; ANE-targeted exports must compile in fp16.
    compute_precision = ct.precision(ct.precision.FLOAT16.value)
    if args.precision == "fp32":
        compute_precision = ct.precision(ct.precision.FLOAT32.value)

    compile_specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=compute_precision,
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        model_type=coreml_model_type,
    )

    # Always keep dequantize_affine on CPU. The matching quantize_affine /
    # choose_qparams_affine ops are already CoreML-unsupported and skipped, so
    # the int8 weight constants get lifted as subgraph inputs. CoreML's
    # dequantize_affine handler then sees `int_data.val == None` and crashes
    # with `AttributeError: 'NoneType' object has no attribute 'shape'`.
    # Running dequant on CPU keeps only the fp16 result inside CoreML.
    skip_ops_for_coreml_delegation = ["torchao.dequantize_affine.default"]
    if args.skip_embedding_delegation:
        skip_ops_for_coreml_delegation += [
            "quantized_decomposed.embedding_4bit.dtype",
            "aten.embedding.default",
        ]

    return CoreMLPartitioner(
        compile_specs=compile_specs,
        take_over_mutable_buffer=False,
        skip_ops_for_coreml_delegation=skip_ops_for_coreml_delegation,
    )


def _finalize_export(export_config: ExportConfig, ep) -> Path:
    """Lower the ExportedProgram to CoreML and write the .pte file."""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
    from executorch.exir.passes.sym_shape_eval_pass import (
        ConstraintBasedSymShapeEvalPass,
    )

    args = export_config.args
    timing_info = export_config.timing_info

    start_time = time.time()
    ep = ep.run_decompositions({})
    timing_info["run_decompositions"] = time.time() - start_time

    partitioner = _build_coreml_partitioner(args)

    logger.info("Lowering to CoreML CPU_AND_NE delegate...")
    start_time = time.time()
    edge_manager = to_edge_transform_and_lower(
        ep,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        partitioner=[partitioner],
    )
    timing_info["to_edge_transform_and_lower"] = time.time() - start_time

    start_time = time.time()
    et_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            # `do_quant_fusion_and_const_prop` covers the older
            # quantized_decomposed.* ops but doesn't fold the torchao.* variants
            # we keep on CPU (quantize_affine, choose_qparams_affine, and our
            # explicitly-skipped dequantize_affine). Without an out variant
            # registered for those ops, the default `to_out_var_pass` raises:
            #   RuntimeError: Missing out variants:
            #     {'torchao::dequantize_affine', 'torchao::choose_qparams_affine',
            #      'torchao::quantize_affine'}
            # Let the lowering finish; ExecuTorch falls back to the functional
            # form at runtime for these ops.
            to_out_var_pass=ToOutVarPass(ignore_to_out_var_failure=True),
        )
    )
    timing_info["to_executorch"] = time.time() - start_time

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with open(output_path, "wb") as f:
        et_program.write_to_file(f)

    if et_program._tensor_data:
        tensor_data_dir = str(output_path.parent)
        et_program.write_tensor_data_to_file(tensor_data_dir)
        logger.info(f"Tensor data written to: {tensor_data_dir}")
    timing_info["Write to file"] = time.time() - start_time

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"CoreML PTE exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def print_timing_summary(timing_info: Dict[str, float], total_time: float) -> None:
    logger.info("=" * 60)
    logger.info("Timing summary:")
    for name, elapsed in timing_info.items():
        logger.info(f"  {name}: {elapsed:.4f} seconds")
    logger.info(f"  Total: {total_time:.4f} seconds")
    logger.info("=" * 60)


def main():
    """
    Main entry point for exporting the Gemma 4 static transformer.

    Orchestrates the export process through three phases:
    1. Build export configuration (model loading, example inputs)
    2. Export model (torch.export.export -> ExportedProgram)
    3. Finalize export (CoreML lowering, to_executorch, file writing)
    """
    overall_start_time = time.time()

    args = parse_args()

    logger.info("Exporting gemma 4 backbone (text decoder)...")
    export_config = _build_export_config(args)
    ep = _export_model(export_config)
    _finalize_export(export_config, ep)

    total_time = time.time() - overall_start_time
    print_timing_summary(export_config.timing_info, total_time)


if __name__ == "__main__":
    main()
