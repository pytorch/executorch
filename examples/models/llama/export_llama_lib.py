# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import json
import logging
import re
import shlex
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, List, Optional, Union

import pkg_resources
import torch

from executorch.devtools.etrecord import generate_etrecord

from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import (
    get_coreml_partitioner,
    get_mps_partitioner,
    get_qnn_partitioner,
    get_vulkan_partitioner,
    get_xnnpack_partitioner,
)

from executorch.extension.llm.export.quantizer_lib import (
    get_coreml_quantizer,
    get_pt2e_quantization_params,
    get_pt2e_quantizers,
    get_qnn_quantizer,
    get_vulkan_quantizer,
)
from executorch.util.activation_memory_profiler import generate_memory_trace

from ..model_factory import EagerModelFactory
from .source_transformation.apply_spin_quant_r1_r2 import (
    fuse_layer_norms,
    get_model_with_r1_r2,
)

from .source_transformation.attention import replace_attention_to_attention_sha
from .source_transformation.quantize import (
    get_quant_embedding_transform,
    get_quant_weight_transform,
)
from .source_transformation.quantized_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
    replace_kv_cache_with_quantized_kv_cache,
)
from .source_transformation.rms_norm import replace_rms_norm_with_native_rms_norm

from .source_transformation.rope import materialze_broadcast_of_rope_freq_cis
from .source_transformation.sdpa import (
    replace_causal_mask,
    replace_kv_cache_with_coreml_kv_cache,
    replace_kv_cache_with_simple_kv_cache,
    replace_sdpa_with_coreml_sdpa,
    replace_sdpa_with_custom_op,
    replace_sdpa_with_flex_sdpa,
    replace_sdpa_with_simple_sdpa,
)
from .source_transformation.vulkan_rope import replace_with_vulkan_rotary_emb

IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


EXECUTORCH_DEFINED_MODELS = ["stories110m", "llama2", "llama3", "llama3_1", "llama3_2"]
TORCHTUNE_DEFINED_MODELS = ["llama3_2_vision"]


class WeightType(Enum):
    LLAMA = "LLAMA"
    FAIRSEQ2 = "FAIRSEQ2"


def set_pkg_name(name: str) -> None:
    global pkg_name
    pkg_name = name


def get_resource_path(resource_name) -> str:
    return pkg_resources.resource_filename(pkg_name, resource_name)


def set_verbosity(val):
    global verbosity_setting
    verbosity_setting = val


def verbose_export():
    return verbosity_setting


def build_model(
    modelname: str = "llama3",
    extra_opts: str = "",
    *,
    par_local_output: bool = False,
    resource_pkg_name: str = __name__,
) -> str:
    if False:  # par_local_output:
        output_dir_path = "par:."
    else:
        output_dir_path = "."

    argString = f"--model {modelname} --checkpoint par:model_ckpt.pt --params par:model_params.json {extra_opts} --output-dir {output_dir_path}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    # pkg_name = resource_pkg_name
    return export_llama(args)


def build_args_parser() -> argparse.ArgumentParser:
    ckpt_dir = f"{Path(__file__).absolute().parent.as_posix()}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    # parser.add_argument(
    #     "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    # )
    parser.add_argument(
        "--model",
        default="llama3",
        choices=EXECUTORCH_DEFINED_MODELS + TORCHTUNE_DEFINED_MODELS,
        help="The Lllama model to export. stories110M, llama2, llama3, llama3_1, and llama3_2 use the same underlying LlamaTransformer architecture defined in ExecuTorch. All other models use TorchTune model definitions.",
    )
    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )
    parser.add_argument(
        "--pt2e_quantize",
        default=None,
        choices=[
            "xnnpack_dynamic",
            "xnnpack_dynamic_qc4",
            "qnn_8a8w",
            "qnn_16a16w",
            "qnn_16a4w",
            "coreml_c4w",
            "coreml_8a_c8w",
            "coreml_8a_c4w",
            "coreml_baseline_8a_c8w",
            "coreml_baseline_8a_c4w",
            "vulkan_8w",
        ],
        help="Use PT2E quantization. Comma separated options. e.g. xnnpack_dynamic (for per channel 8 bit weight), xnnpack_dynamic_qc4 (for per channel 4 bit weight), embedding.",
    )

    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=_qmode_type,
        default=None,
        help="type of quantization",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=f"{ckpt_dir}/params/demo_rand_params.pth",
        help="checkpoint path",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="checkpoint directory. Use with a sharded checkpoint, not for the standard llama2 model. Note, checkpoint_dir takes precedence over checkpoint if both are set.",
    )

    parser.add_argument(
        "--use_qnn_sha",
        action="store_true",
        help="Change multi head attention to multiple single head attention for qnn backend (Qualcomm)",
    )

    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=None,
        help="Sequence length for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="Once upon a time",
        help="Calibration prompts from users",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        default=None,
        help="tokenizer path (Note: .model not .bin)",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )
    parser.add_argument(
        "--quantize_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using int8 per token quantized kv cache",
    )
    parser.add_argument(
        "--num_sharding",
        type=int,
        default=0,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
    )
    parser.add_argument(
        "--use_sdpa_with_kv_cache",
        default=False,
        action="store_true",
        help="Whether to use sdpa_with_kv_cache update op when using kv cache",
    )
    parser.add_argument(
        "--disable_dynamic_shape",
        dest="enable_dynamic_shape",
        default=True,  # Enable this by default
        action="store_false",
        help="Enable dynamic shape along seq dim. Used for faster prefill",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=f"{ckpt_dir}/params/demo_config.json",
        help="config.json",
    )
    parser.add_argument(
        "--optimized_rotation_path",
        default=None,
        required=False,
        help="[QNN backend] Optimized rotation checkpoint path. Just apply R1/R2 here."
        "You can download the optimized rotation matrices from https://github.com/facebookresearch/SpinQuant/tree/main",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        help='metadata string in json format. Example {"key": 1, "key2": "value2"}',
    )
    parser.add_argument(
        "-s",
        "--so_library",
        default=None,
        required=False,
        help="shared library for quantized operators",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    parser.add_argument(
        "-prof",
        "--profile_path",
        default=None,
        help="Use cProfile to profile model export. Results saved to profile_path as a html file.",
    )
    parser.add_argument(
        "-G",
        "--group_size",
        type=int,
        default=None,
        help="group_size for weight quantization",
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Override the dtype of the model (default is the checkpoint dtype)."
        "Options: fp32, fp16, bf16. Please be aware that only some backends support fp16 and bf16.",
    )

    parser.add_argument(
        "-n",
        "--output_name",
        default=None,
        help="Override the output filename of the saved pte model file.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )

    parser.add_argument("-2", "--fairseq2", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-X",
        "--xnnpack",
        action="store_true",
        help="Delegate to DQLinear ops to the xnnpack backend",
    )
    parser.add_argument(
        "--xnnpack-extended-ops",
        action="store_true",
        help="Delegate more operators beyond DQLinear to the xnnpack backend. Requires -X or --xnnpack to be set.",
    )
    parser.add_argument("-V", "--vulkan", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument(
        "--coreml-enable-state",
        action="store_true",
        help="This option is only for coreml, and is only supported for MacOS15+/iOS18+",
    )
    parser.add_argument(
        "--coreml-preserve-sdpa",
        action="store_true",
        help="This option is only for coreml: Preserve sdpa in torch edge program to use coreml iOS18.sdpa op",
    )
    parser.add_argument(
        "--coreml-quantize",
        default=None,
        choices=["b4w", "c4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight), c4w (for channelwise 4 bit weight)",
    )
    parser.add_argument(
        "--coreml-ios",
        type=int,
        default=15,
        choices=(15, 16, 17, 18),
        help="This option is only for coreml: The minimum iOS version to deploy",
    )
    parser.add_argument(
        "--coreml-compute-units",
        type=str,
        default="cpu_only",
        choices=("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"),
        help="This option is only for coreml: the compute units to use when running the model",
    )
    parser.add_argument(
        "--qnn",
        action="store_true",
        help="Delegate llama2 to qnn backend (Qualcomm), please use it --kv_cahce=True",
    )

    parser.add_argument(
        "--expand_rope_table",
        default=False,
        action="store_true",
        help="[Temp workaround] Expand sin/cos table in head dim to take vectorized path in optimized kernels.",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate the ETRecord debug artifact.",
    )

    parser.add_argument(
        "--generate_full_logits",
        action="store_true",
        required=False,
        default=False,
        help="Generate logits for all inputs.",
    )

    parser.add_argument(
        "--soc_model",
        help="[QNN backend] SoC model of current device. e.g. 'SM8650' for Snapdragon 8 Gen 3.",
        type=str,
        required=False,
        default="SM8650",
    )

    parser.add_argument(
        "-sq",
        "--use_spin_quant",
        type=str,
        default=None,
        choices=["cuda", "native"],
        help="Use SpinQuant for better quantization performance. Only support cuda and native.",
    )

    parser.add_argument(
        "-qat",
        "--use_qat",
        default=False,
        action="store_true",
        help="Whether the checkpoin is pre-quantized with QAT or not.",
    )

    parser.add_argument(
        "-lora",
        "--use_lora",
        type=int,
        default=0,
        help="Whether the checkpoint contains LoRA adaptors or not. 0: no LoRA adaptors; "
        "otherwise, it means the rank of LoRA adaptors. Currently it only works if QAT is enabled.",
    )

    parser.add_argument(
        "--preq_mode",
        type=str,
        default=None,
        choices=["8da4w", "8da4w_output_8da8w"],
        help="Quantization mode used for pre-quantized checkpoint. Only support 8da4w and 8da4w_output_8da8w right now.",
    )

    parser.add_argument(
        "--preq_group_size",
        type=int,
        default=32,
        help="group_size for pre-quantized checkpoint weight quantization",
    )

    parser.add_argument(
        "--preq_embedding_quantize",
        default="8,0",
        type=str,
        help="type of embedding quantization for pre-quantized checkpoint, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )

    parser.add_argument(
        "--use_attention_sink",
        default=None,
        type=str,
        help="Use attention sink to have fluent multi-round conversation. '<sink_size>,<window_size>,<batch_eviction_size>', e.g., '4,2044,1024'.",
    )

    parser.add_argument(
        "--output_prune_map",
        default=None,
        help="path to the output pruning token mapping file (token_map.json)",
    )

    parser.add_argument(
        "--input_prune_map",
        default=None,
        help="path to the input pruning token mapping file (token_map.json)",
    )

    parser.add_argument(
        "--export_only",
        default=False,
        action="store_true",
        help="If true, stops right after torch.export() and saves the exported model.",
    )
    return parser


def canonical_path(path: Union[str, Path], *, dir: bool = False) -> str:
    path = str(path)

    if verbose_export():
        print(f"creating canonical path for {path}")

    if not path.startswith("par:"):
        return path

    if not IS_FBCODE:
        print("not FBCODE")
        return path[4:]
    else:
        return_val = pkg_resources.resource_filename(pkg_name, path[4:])
        if verbose_export():
            print(f"canonical name is: {return_val}")
        return return_val


def export_llama(args) -> str:
    if args.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(args.profile_path):
                builder = _export_llama(args)
                assert (
                    filename := builder.get_saved_pte_filename()
                ) is not None, "Fail to get file name from builder"
                return filename
        except ImportError:
            print(
                "Please run `pip install snakeviz` to install required dependencies for cProfiler flamegraph."
            )
            return ""
    else:
        builder = _export_llama(args)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(args) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """
    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint) if args.checkpoint else None
    checkpoint_dir = (
        canonical_path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    params_path = canonical_path(args.params)
    output_dir_path = canonical_path(args.output_dir, dir=True)
    weight_type = WeightType.FAIRSEQ2 if args.fairseq2 else WeightType.LLAMA

    # dtype override
    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
    elif args.quantization_mode in ["8da4w", "8da4w-gptq"]:
        dtype_override = DType["fp16"]
    else:
        dtype_override = None

    return (
        _load_llama_model(
            args.model,
            checkpoint=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            params_path=params_path,
            use_kv_cache=args.use_kv_cache,
            use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
            generate_full_logits=args.generate_full_logits,
            weight_type=weight_type,
            enable_dynamic_shape=args.enable_dynamic_shape,
            calibration_tasks=args.calibration_tasks,
            calibration_limit=args.calibration_limit,
            calibration_seq_length=args.calibration_seq_length,
            calibration_data=args.calibration_data,
            tokenizer_path=args.tokenizer_path,
            verbose=args.verbose,
            max_seq_len=args.max_seq_length,
            input_prune_map_path=args.input_prune_map,
            output_prune_map_path=args.output_prune_map,
            metadata_str=args.metadata,
            dtype_override=dtype_override,
            args=args,
        )
        .set_output_dir(output_dir_path)
        .source_transform(_get_source_transforms(args.model, dtype_override, args))
    )


def get_quantizer_and_quant_params(args):
    pt2e_quant_params = get_pt2e_quantization_params(
        args.pt2e_quantize, args.quantization_mode
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, args.so_library)
    quant_dtype = None
    if args.qnn and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack and qnn"
        qnn_quantizer, quant_dtype = get_qnn_quantizer(
            args.pt2e_quantize, args.quantization_mode
        )
        quantizers.append(qnn_quantizer)
    if args.coreml and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack / qnn and coreml"
        coreml_quantizer = get_coreml_quantizer(args.pt2e_quantize)
        quantizers.append(coreml_quantizer)
    if args.vulkan and args.pt2e_quantize:
        assert (
            len(quantizers) == 0
        ), "Should not enable both vulkan and other quantizers"
        vulkan_quantizer = get_vulkan_quantizer(args.pt2e_quantize)
        quantizers.append(vulkan_quantizer)
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


def _qmode_type(value):
    choices = ["int8", "8da4w", "8da4w-gptq", "vulkan_4w"]
    patterns = [r"torchao:8da(\d+)w", r"torchao:fpa(\d+)w"]

    if value in choices:
        return value

    for pattern in patterns:
        matches = re.findall(pattern, value)
        if len(matches) == 1:
            return value

    raise argparse.ArgumentTypeError(
        f"Got qmode {value}, but expected one of {choices}, or one of the regex patterns {patterns}."
    )


def _validate_args(args):
    """
    TODO: Combine all the backends under --backend args
    """
    if args.enable_dynamic_shape and (args.coreml or args.mps or args.qnn):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends."
            " Please use --disable_dynamic_shape."
        )

    if args.num_sharding > 0 and not args.qnn:
        raise ValueError("Model shard is only supported with qnn backend now.")

    if (
        args.quantization_mode is not None
        and args.quantization_mode.startswith("torchao:")
    ) or (
        args.embedding_quantize is not None
        and args.embedding_quantize.startswith("torchao:")
    ):
        if args.enable_dynamic_shape:
            raise ValueError(
                "Dynamic shape is not currently supported with torchao ops. Please use --disable_dynamic_shape."
                "If you need this feature, please file an issue."
            )


def _export_llama(args) -> LLMEdgeManager:  # noqa: C901
    _validate_args(args)
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    # export_to_edge
    builder_exported = _prepare_for_llama_export(args).export()

    if args.export_only:
        exit()

    builder_exported_to_edge = builder_exported.pt2e_quantize(
        quantizers
    ).export_to_edge()

    modelname = builder_exported_to_edge.modelname

    # to_backend
    partitioners = []

    # Order matters here, dynamic quantization should be applied first when both xnnpack and xnnpack_extended_ops are enabled
    if (
        pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None
    ) or (args.xnnpack):
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=True)
        )

        # force xnnpack to be true if pt2e_quant_params is not None and args.xnnpack is False
        args.xnnpack = True
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack_extended_ops:
        assert args.xnnpack, "xnnpack_extended_ops requires xnnpack to be enabled"
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False)
        )
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        partitioners.append(
            get_vulkan_partitioner(
                args.dtype_override,
                args.enable_dynamic_shape,
            )
        )
        # Apply XNNPACK after Vulkan so that undelegated ops can be accelerated by XNNPACK
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False)
        )
        modelname = f"vulkan_{modelname}"

    if args.mps:
        partitioners.append(get_mps_partitioner(args.use_kv_cache))
        modelname = f"mps_{modelname}"

    if args.coreml:
        coreml_partitioner = get_coreml_partitioner(
            args.coreml_ios,
            args.embedding_quantize,
            args.pt2e_quantize,
            args.coreml_quantize,
            args.coreml_compute_units,
        )
        partitioners.append(coreml_partitioner)
        modelname = f"coreml_{modelname}"

    if args.qnn:
        from executorch.extension.llm.custom_ops import model_sharding

        partitioners.append(
            get_qnn_partitioner(
                args.use_kv_cache, args.pt2e_quantize, args.num_sharding, args.soc_model
            )
        )
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
        from executorch.backends.qualcomm.utils.utils import _transform, tag_quant_io

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`, Optional type has no attribute `exported_program`
        _transform(builder_exported_to_edge.edge_manager.exported_program())

        if args.num_sharding > 0:
            model_sharding.split_graph(
                builder_exported_to_edge.edge_manager.exported_program(),
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=args.num_sharding,
            )

        from functools import partial

        # pyre-ignore
        from executorch.backends.qualcomm.quantizer.custom_annotation import (
            get_custom_quant_ios_dtype,
        )

        atten = builder_exported_to_edge.model.layers[0].attention
        if args.use_qnn_sha:
            cache_shape = torch.Size(
                (atten.max_batch_size, atten.max_seq_len, atten.head_dim)
            )
        else:
            cache_shape = torch.Size(
                (
                    atten.max_batch_size,
                    atten.max_seq_len,
                    atten.n_kv_heads,
                    atten.head_dim,
                )
            )
        # pyre-ignore
        tag_quant_io(
            builder_exported_to_edge.edge_manager.exported_program().graph_module,
            partial(get_custom_quant_ios_dtype, cache_shape),  # pyre-ignore
        )

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            # pyre-fixme[16]: Module `backends` has no attribute `qualcomm`.
            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord(
                et_record="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            # pyre-fixme[16]: Module `backends` has no attribute `qualcomm`.
            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

    if args.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if args.output_name:
        modelname = args.output_name
        if modelname.endswith(".pte"):
            output_file = modelname
            modelname = modelname[:-4]
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
        else:
            output_file = f"{builder.output_dir}/{modelname}.pte"
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
    else:
        output_file = f"{builder.output_dir}/{modelname}.pte"

    builder.save_to_pte(output_file)

    return builder


def _load_llama_model_metadata(
    weight_type: WeightType,
    use_kv_cache: bool,
    use_sdpa_with_kv_cache: bool,
    enable_dynamic_shape: bool,
    max_seq_len: int,
    n_layers: int,
    vocab_size: int,
    metadata_str: Optional[str] = None,
):
    is_fairseq2 = weight_type == WeightType.FAIRSEQ2
    metadata = {
        "get_bos_id": 3 if is_fairseq2 else 1,
        "get_eos_ids": [3] if is_fairseq2 else [2],
        "get_max_seq_len": max_seq_len,
        "get_n_layers": n_layers,
        "get_vocab_size": vocab_size,
        "use_kv_cache": use_kv_cache,
        "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
        "enable_dynamic_shape": enable_dynamic_shape,
    }
    if metadata_str:
        try:
            extra = json.loads(metadata_str)
            for k, v in extra.items():
                metadata[k] = v
        except JSONDecodeError:
            logging.error("Invalid metadata, should be a valid JSON string")
    return metadata


def _load_llama_model(
    modelname: str = "llama3",
    *,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    params_path: str,
    use_kv_cache: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    generate_full_logits: bool = False,
    weight_type: WeightType = WeightType.LLAMA,
    enable_dynamic_shape: bool = False,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    calibration_data: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    verbose: bool = False,
    max_seq_len: int = 128,
    input_prune_map_path: Optional[str] = None,
    output_prune_map_path: Optional[str] = None,
    metadata_str: Optional[str] = None,
    dtype_override: Optional[DType] = None,
    args,
) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """

    assert (
        checkpoint or checkpoint_dir
    ) and params_path, "Both checkpoint/checkpoint_dir and params can't be empty"
    logging.info(
        f"Loading model with checkpoint={checkpoint}, params={params_path}, use_kv_cache={use_kv_cache}, weight_type={weight_type}"
    )

    if modelname in EXECUTORCH_DEFINED_MODELS:
        module_name = "llama"
        model_class_name = "Llama2Model"  # TODO: Change to "LlamaModel" in examples/models/llama/model.py.
    elif modelname in TORCHTUNE_DEFINED_MODELS:
        if modelname == "llama3_2_vision":
            module_name = "llama3_2_vision"
            model_class_name = "Llama3_2Decoder"
        else:
            raise ValueError(f"{modelname} is not a valid Llama model.")
    else:
        raise ValueError(f"{modelname} is not a valid Llama model.")

    model, example_inputs, example_kwarg_inputs, dynamic_shapes = (
        EagerModelFactory.create_model(
            module_name,
            model_class_name,
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            params=params_path,
            use_kv_cache=use_kv_cache,
            use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
            generate_full_logits=generate_full_logits,
            fairseq2=weight_type == WeightType.FAIRSEQ2,
            max_seq_len=max_seq_len,
            enable_dynamic_shape=enable_dynamic_shape,
            input_prune_map_path=input_prune_map_path,
            output_prune_map_path=output_prune_map_path,
            args=args,
        )
    )
    if dtype_override:
        assert isinstance(
            dtype_override, DType
        ), "Override dtype needs to be of type <DType>"
        torch_dtype = dtype_override.to_torch_dtype()
        logging.info(f"model.to {torch_dtype}")
        model = model.to(dtype=torch_dtype)
        dtype = dtype_override
    else:
        state_dict = model.state_dict()
        dtype = state_dict[next(iter(state_dict))].dtype
        assert dtype in [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ], f"Only support bfloat16, fp16 or fp32 got {dtype}"
        logging.info(f"Loaded model with dtype={dtype}")

        if dtype == torch.bfloat16:
            dtype = DType.bf16
        elif dtype == torch.float16:
            dtype = DType.fp16
        elif dtype == torch.float32:
            dtype = DType.fp32
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.max_seq_len,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
        generate_full_logits=generate_full_logits,
        example_inputs=example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        enable_dynamic_shape=enable_dynamic_shape,
        calibration_tasks=calibration_tasks,
        calibration_limit=calibration_limit,
        calibration_seq_length=calibration_seq_length,
        calibration_data=calibration_data,
        tokenizer_path=tokenizer_path,
        verbose=verbose,
        metadata=_load_llama_model_metadata(
            weight_type,
            use_kv_cache,
            use_sdpa_with_kv_cache,
            enable_dynamic_shape,
            # pyre-fixme[6]: For 5th argument expected `ModelArgs` but got
            #  `Union[Tensor, Module]`.
            model.max_seq_len,
            # pyre-fixme[6]: For 6th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.n_layers,
            # pyre-fixme[6]: For 7th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.vocab_size,
            metadata_str,
        ),
        args=args,
    )


def _get_source_transforms(  # noqa
    modelname: str, dtype_override: Optional[DType], args
) -> List[Callable[[torch.nn.Module], torch.nn.Module]]:
    transforms = []

    if args.use_spin_quant:
        if args.use_spin_quant == "cuda":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_cuda_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_cuda_for_spin_quant)
        elif args.use_spin_quant == "native":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_native_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_native_for_spin_quant)

    if args.quantization_mode:
        """
        When this option is selected, it finds all linear layers and transforms
        into quantized linear equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        if quantization_mode is enabled, it will quantize any remaining linear
        ops that is not quantized.

        There are cases where this may be a no-op, namely, if all linears are
        quantized in the checkpoint.
        """
        modelname = f"{modelname}_q"
        transforms.append(
            get_quant_weight_transform(args, dtype_override, verbose_export())
        )

    if args.embedding_quantize:
        """
        When this option is selected, it finds all embedding layers and transforms
        into quantized embedding equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        this wil be a no-op.
        """
        modelname = f"{modelname}_e"
        transforms.append(get_quant_embedding_transform(args))

    if args.expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    if args.use_sdpa_with_kv_cache:
        transforms.append(replace_kv_cache_with_custom_kv_cache)
        transforms.append(replace_sdpa_with_custom_op)

    if args.quantize_kv_cache:
        assert args.use_kv_cache, "quantize_kv_cache requires use_kv_cache=True"
        transforms.append(replace_kv_cache_with_quantized_kv_cache)

    if args.use_kv_cache:
        if args.qnn:
            from executorch.backends.qualcomm.utils.utils import (
                convert_linear_to_conv2d,
            )

            if args.use_qnn_sha:
                if args.optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(
                        get_model_with_r1_r2(args.optimized_rotation_path)
                    )
                transforms.append(replace_attention_to_attention_sha)
                transforms.append(replace_causal_mask)
                transforms.append(replace_rms_norm_with_native_rms_norm)
                # pyre-fixme[16]: Module `backends` has no attribute `qualcomm`.
                transforms.append(convert_linear_to_conv2d)
            else:
                transforms.append(replace_kv_cache_with_simple_kv_cache)
                transforms.append(replace_sdpa_with_flex_sdpa)
                transforms.append(replace_causal_mask)
                transforms.append(replace_rms_norm_with_native_rms_norm)
                if args.optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(
                        get_model_with_r1_r2(args.optimized_rotation_path)
                    )
                # pyre-fixme[16]: Module `backends` has no attribute `qualcomm`.
                transforms.append(convert_linear_to_conv2d)

        elif args.mps:
            # Currently mps doesn't support sdpa op, use the simpler decomposition
            # to get free perf gain.
            transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_causal_mask)

        elif args.coreml:
            # iOS 18 introduced fused sdpa op
            if args.coreml_ios >= 18:
                transforms.append(replace_sdpa_with_coreml_sdpa)
            else:
                transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_kv_cache_with_coreml_kv_cache)

    if args.vulkan:
        transforms.append(replace_with_vulkan_rotary_emb)

    return transforms
