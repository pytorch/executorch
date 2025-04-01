# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
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
from functools import partial
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, List, Optional, Union

import pkg_resources
import torch

from executorch.backends.vulkan._passes.remove_asserts import remove_asserts
from executorch.devtools.backend_debug import print_delegation_info

from executorch.devtools.etrecord import generate_etrecord
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

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
from omegaconf import DictConfig, OmegaConf

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


# All models that leverage the transformer architecture defined in llama_transformer.py.
EXECUTORCH_DEFINED_MODELS = [
    "stories110m",
    "llama2",
    "llama3",
    "llama3_1",
    "llama3_2",
    "static_llama",
    "qwen2_5",
    "phi_4_mini",
    "smollm2",
]
TORCHTUNE_DEFINED_MODELS = ["llama3_2_vision"]
HUGGING_FACE_REPO_IDS = {
    "qwen2_5": "Qwen/Qwen2.5-1.5B",
    "phi_4_mini": "microsoft/Phi-4-mini-instruct",
    "smollm2": "HuggingFaceTB/SmolLM-135M",
}


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
    model: str,
    checkpoint: str,
    params: str,
    output_dir: Optional[str] = ".",
    extra_opts: Optional[str] = "",
) -> str:
    argString = f"--model {model} --checkpoint {checkpoint} --params {params} {extra_opts} --output-dir {output_dir}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    return export_llama(args)


def build_args_parser() -> argparse.ArgumentParser:
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
        "--use_shared_embedding",
        action="store_true",
        help="Whether the embedding/unembedding weights should be shared.  Only available with torchao kernels.",
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
        required=False,
        help="Path to the checkpoint .pth file. When not provided, the model will be initialized with random weights.",
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
        required=False,
        help="Config file for model parameters. When not provided, the model will fallback on default values defined in examples/models/llama/model_args.py.",
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
        help="Provide the dtype of the model. This must match up with the supported dtypes of the backends that you are using."
        "Please be aware that only some backends support fp16 and bf16.",
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

    parser.add_argument(
        "--max_context_length",
        type=int,
        default=128,
        help="maximum length of context for model to remember",
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


def get_default_llm_config() -> DictConfig:
    default_args = build_args_parser().parse_args([])
    return _convert_args_to_config(default_args)


def _convert_args_to_config(args: argparse.Namespace) -> DictConfig:
    """Convert argparse.Namespace to DictConfig."""
    # Create a dictionary from args
    args_dict = {}

    # Add model settings
    args_dict["model"] = {
        "name": args.model,
        "type": "LLAMA" if not args.fairseq2 else "FAIRSEQ2",
        "dtype_override": args.dtype_override if "dtype_override" in args else None,
        "params": args.params if "params" in args else None,
        "checkpoint": args.checkpoint if "checkpoint" in args else None,
        "checkpoint_dir": args.checkpoint_dir if "checkpoint_dir" in args else None,
        "tokenizer_path": args.tokenizer_path if "tokenizer_path" in args else None,
        "metadata": args.metadata if "metadata" in args else None,
        "use_shared_embedding": (
            args.use_shared_embedding if "use_shared_embedding" in args else None
        ),
    }

    # Add export settings
    args_dict["export"] = {
        "output_dir": args.output_dir if "output_dir" in args else None,
        "output_name": args.output_name if "output_name" in args else None,
        "so_library": args.so_library if "so_library" in args else None,
        "export_only": args.export_only if "export_only" in args else None,
    }

    # Add sequence settings
    args_dict["sequence"] = {
        "max_seq_length": args.max_seq_length if "max_seq_length" in args else None,
        "max_context_length": (
            args.max_context_length if "max_context_length" in args else None
        ),
    }

    # Add KV cache settings
    args_dict["kv_cache"] = {
        "use_kv_cache": args.use_kv_cache if "use_kv_cache" in args else None,
        "quantize_kv_cache": (
            args.quantize_kv_cache if "quantize_kv_cache" in args else None
        ),
        "use_sdpa_with_kv_cache": (
            args.use_sdpa_with_kv_cache if "use_sdpa_with_kv_cache" in args else None
        ),
    }

    # Add quantization settings
    args_dict["quantization"] = {
        "mode": args.quantization_mode if "quantization_mode" in args else None,
        "embedding_quantize": (
            args.embedding_quantize if "embedding_quantize" in args else None
        ),
        "pt2e_quantize": args.pt2e_quantize if "pt2e_quantize" in args else None,
        "group_size": args.group_size if "group_size" in args else None,
        "use_spin_quant": args.use_spin_quant if "use_spin_quant" in args else None,
        "use_qat": args.use_qat if "use_qat" in args else None,
        "use_lora": args.use_lora if "use_lora" in args else None,
        "preq_mode": args.preq_mode if "preq_mode" in args else None,
        "preq_group_size": args.preq_group_size if "preq_group_size" in args else None,
        "preq_embedding_quantize": (
            args.preq_embedding_quantize if "preq_embedding_quantize" in args else None
        ),
    }

    # Add calibration settings
    args_dict["calibration"] = {
        "tasks": args.calibration_tasks if "calibration_tasks" in args else None,
        "limit": args.calibration_limit if "calibration_limit" in args else None,
        "seq_length": (
            args.calibration_seq_length if "calibration_seq_length" in args else None
        ),
        "data": args.calibration_data if "calibration_data" in args else None,
    }

    # Add backend settings
    args_dict["backend"] = {
        "xnnpack": {
            "enabled": args.xnnpack if "xnnpack" in args else None,
            "extended_ops": (
                args.xnnpack_extended_ops if "xnnpack_extended_ops" in args else None
            ),
        },
        "coreml": {
            "enabled": args.coreml if "coreml" in args else None,
            "enable_state": (
                args.coreml_enable_state if "coreml_enable_state" in args else None
            ),
            "preserve_sdpa": (
                args.coreml_preserve_sdpa if "coreml_preserve_sdpa" in args else None
            ),
            "quantize": args.coreml_quantize if "coreml_quantize" in args else None,
            "ios": args.coreml_ios if "coreml_ios" in args else None,
            "compute_units": (
                args.coreml_compute_units if "coreml_compute_units" in args else None
            ),
        },
        "vulkan": {
            "enabled": args.vulkan if "vulkan" in args else None,
        },
        "qnn": {
            "enabled": args.qnn if "qnn" in args else None,
            "use_sha": args.use_qnn_sha if "use_qnn_sha" in args else None,
            "soc_model": args.soc_model if "soc_model" in args else None,
            "optimized_rotation_path": (
                args.optimized_rotation_path
                if "optimized_rotation_path" in args
                else None
            ),
        },
        "mps": {
            "enabled": args.mps if "mps" in args else None,
        },
    }

    # Add additional settings
    args_dict["misc"] = {
        "profile_memory": args.profile_memory if "profile_memory" in args else None,
        "profile_path": args.profile_path if "profile_path" in args else None,
        "enable_dynamic_shape": (
            args.enable_dynamic_shape if "enable_dynamic_shape" in args else None
        ),
        "num_sharding": args.num_sharding if "num_sharding" in args else None,
        "expand_rope_table": (
            args.expand_rope_table if "expand_rope_table" in args else None
        ),
        "generate_etrecord": (
            args.generate_etrecord if "generate_etrecord" in args else None
        ),
        "generate_full_logits": (
            args.generate_full_logits if "generate_full_logits" in args else None
        ),
        "use_attention_sink": (
            args.use_attention_sink if "use_attention_sink" in args else None
        ),
        "output_prune_map": (
            args.output_prune_map if "output_prune_map" in args else None
        ),
        "input_prune_map": args.input_prune_map if "input_prune_map" in args else None,
        "verbose": args.verbose if "verbose" in args else None,
    }

    # Convert to DictConfig
    return OmegaConf.create(args_dict)


def export_llama(args: Union[argparse.Namespace, DictConfig]) -> str:
    """Export Llama model to flatbuffer format."""
    # Convert args to config if needed
    if isinstance(args, argparse.Namespace):
        config = _convert_args_to_config(args)
    else:
        config = args

    # If a checkpoint isn't provided for an HF OSS model, download and convert the
    # weights first.
    if not config.model.checkpoint and config.model.name in HUGGING_FACE_REPO_IDS:
        repo_id = HUGGING_FACE_REPO_IDS[config.model.name]
        if config.model.name == "qwen2_5":
            from executorch.examples.models.qwen2_5 import (  # pyre-ignore
                convert_weights,
            )
        elif config.model.name == "phi_4_mini":
            from executorch.examples.models.phi_4_mini import (  # pyre-ignore
                convert_weights,
            )
        elif config.model.name == "smollm2":
            from executorch.examples.models.smollm2 import (  # pyre-ignore
                convert_weights,
            )
        else:
            raise ValueError(
                f"Converting weights to meta format for {config.model.name} is not yet supported"
            )
        config.model.checkpoint = download_and_convert_hf_checkpoint(
            repo_id, convert_weights
        )

    if config.misc.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(config.misc.profile_path):
                builder = _export_llama(config)
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
        builder = _export_llama(config)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(config: DictConfig) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """
    # load model from checkpoint and params.json
    checkpoint_path = (
        canonical_path(config.model.checkpoint) if config.model.checkpoint else None
    )
    checkpoint_dir = (
        canonical_path(config.model.checkpoint_dir)
        if config.model.checkpoint_dir
        else None
    )
    params_path = canonical_path(config.model.params) if config.model.params else None
    output_dir_path = canonical_path(config.export.output_dir, dir=True)
    weight_type = (
        WeightType.FAIRSEQ2 if config.model.type == "FAIRSEQ2" else WeightType.LLAMA
    )

    # Convert dtype override string arg to actual type.
    dtype_override = DType[config.model.dtype_override]

    calibration_tasks = (
        None
        if config.calibration.tasks is None
        else OmegaConf.to_container(config.calibration.tasks)
    )
    edge_manager = _load_llama_model(
        config.model.name,
        checkpoint=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        params_path=params_path,
        use_kv_cache=config.kv_cache.use_kv_cache,
        use_sdpa_with_kv_cache=config.kv_cache.use_sdpa_with_kv_cache,
        generate_full_logits=config.misc.generate_full_logits,
        weight_type=weight_type,
        enable_dynamic_shape=config.misc.enable_dynamic_shape,
        calibration_tasks=calibration_tasks,
        calibration_limit=config.calibration.limit,
        calibration_seq_length=config.calibration.seq_length,
        calibration_data=config.calibration.data,
        tokenizer_path=config.model.tokenizer_path,
        verbose=config.misc.verbose,
        max_seq_len=config.sequence.max_seq_length,
        max_context_len=config.sequence.max_context_length,
        input_prune_map_path=config.misc.input_prune_map,
        output_prune_map_path=config.misc.output_prune_map,
        metadata_str=config.model.metadata,
        dtype_override=dtype_override,
        config=config,
    )

    # At this point, the model is loaded in the default fp32.

    # Checkpoint dtype should be lower or equal precision to the dtype override.
    checkpoint_dtype = edge_manager.model.checkpoint_dtype
    if not (
        checkpoint_dtype == dtype_override.to_torch_dtype()
        or (
            checkpoint_dtype == torch.float16
            and dtype_override.to_torch_dtype() == torch.float32
        )
        or (
            checkpoint_dtype == torch.bfloat16
            and dtype_override.to_torch_dtype() == torch.float32
        )
    ):
        logging.warning(
            f"Checkpoint dtype {checkpoint_dtype} precision is higher than dtype override {dtype_override.to_torch_dtype()}."
        )

    edge_manager.model = edge_manager.model.to(dtype=dtype_override.to_torch_dtype())

    # We want to quantize (in the source transforms) the weights of the model
    # in the checkpoint dtype.
    logging.info(f"Checkpoint dtype: {edge_manager.model.checkpoint_dtype}")
    edge_manager = edge_manager.set_output_dir(output_dir_path).source_transform(
        _get_source_transforms(
            modelname=config.model.name,
            dtype_override=dtype_override,
            checkpoint_dtype=DType.from_torch_dtype(checkpoint_dtype),
            config=config,
        )
    )

    return edge_manager


def get_quantizer_and_quant_params(config: DictConfig):
    pt2e_quant_params = get_pt2e_quantization_params(
        config.quantization.pt2e_quantize, config.quantization.mode
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, config.export.so_library)
    quant_dtype = None
    if config.backend.qnn.enabled and config.quantization.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack and qnn"
        qnn_quantizer, quant_dtype = get_qnn_quantizer(
            config.quantization.pt2e_quantize, config.quantization.mode
        )
        quantizers.append(qnn_quantizer)
    if config.backend.coreml.enabled and config.quantization.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack / qnn and coreml"
        coreml_quantizer = get_coreml_quantizer(config.quantization.pt2e_quantize)
        quantizers.append(coreml_quantizer)
    if config.backend.vulkan.enabled and config.quantization.pt2e_quantize:
        assert (
            len(quantizers) == 0
        ), "Should not enable both vulkan and other quantizers"
        vulkan_quantizer = get_vulkan_quantizer(config.quantization.pt2e_quantize)
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

    if args.max_context_length < args.max_seq_length:
        raise ValueError(
            f"max_context_length {args.max_context_length} must be >= max_seq_len {args.max_seq_length}. max_context_length impacts kv cache size that is used to remember history, while max_seq_length refers to user prompt length. Please use --max_context_length to specify context length."
        )
    if args.enable_dynamic_shape and (args.coreml or args.mps or args.qnn):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends."
            " Please use --disable_dynamic_shape."
        )

    if args.num_sharding > 0 and not args.qnn:
        raise ValueError("Model shard is only supported with qnn backend now.")

    if args.use_shared_embedding:
        if not (
            args.embedding_quantize is not None
            and args.embedding_quantize.startswith("torchao:")
        ):
            raise ValueError(
                "Shared embedding is only supported with torchao quantization."
            )


def _to_edge_and_lower_llama_xnnpack(
    builder_exported,
    modelname,
    additional_passes,
    pt2e_quant_params,
    quantizers,
    quant_dtype,
    config: DictConfig,
) -> LLMEdgeManager:
    partitioners = []

    # Order matters here, dynamic quantization should be applied first when both xnnpack and xnnpack_extended_ops are enabled
    partitioners.append(get_xnnpack_partitioner(dynamic_quant_only_partitioner=True))

    modelname = f"xnnpack_dq_{modelname}"

    if config.backend.xnnpack.extended_ops:
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False)
        )
        modelname = f"xnnpack_{modelname}"

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    # TODO: Enable generating ETRecord with XNNPack and to_edge_transform_and_lower().
    if config.misc.generate_etrecord:
        raise NotImplementedError(
            "export_llama does not support XNNPack and generating ETRecord at the moment."
        )

    builder = builder_exported.pt2e_quantize(quantizers).to_edge_transform_and_lower(
        partitioners
    )
    if config.misc.verbose:
        print_delegation_info(builder.edge_manager.exported_program().graph_module)

    return builder.to_executorch(passes=additional_passes)


def _to_edge_and_lower_llama(  # noqa: C901
    builder_exported,
    modelname,
    additional_passes,
    pt2e_quant_params,
    quantizers,
    quant_dtype,
    config: DictConfig,
):
    builder_exported_to_edge = builder_exported.pt2e_quantize(
        quantizers
    ).export_to_edge()

    # to_backend
    partitioners = []
    if config.backend.vulkan.enabled:
        partitioners.append(
            get_vulkan_partitioner(
                config.model.dtype_override,
                config.misc.enable_dynamic_shape,
            )
        )
        # Apply XNNPACK after Vulkan so that undelegated ops can be accelerated by XNNPACK
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False)
        )
        modelname = f"vulkan_{modelname}"

        # Need to remove asserts from the graph to prevent graph breaks
        remove_asserts(builder_exported_to_edge.edge_manager.exported_program())

    if config.backend.mps.enabled:
        partitioners.append(get_mps_partitioner(config.kv_cache.use_kv_cache))
        modelname = f"mps_{modelname}"

    if config.backend.coreml.enabled:
        coreml_partitioner = get_coreml_partitioner(
            config.backend.coreml.ios,
            config.quantization.embedding_quantize,
            config.quantization.pt2e_quantize,
            config.backend.coreml.quantize,
            config.backend.coreml.compute_units,
        )
        partitioners.append(coreml_partitioner)
        modelname = f"coreml_{modelname}"

    if config.backend.qnn.enabled:
        from executorch.extension.llm.custom_ops import model_sharding

        partitioners.append(
            get_qnn_partitioner(
                config.kv_cache.use_kv_cache,
                config.quantization.pt2e_quantize,
                config.misc.num_sharding,
                config.backend.qnn.soc_model,
            )
        )
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
        from executorch.backends.qualcomm._passes.annotate_decomposed import (
            AnnotateDecomposed,
        )
        from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY
        from executorch.backends.qualcomm.utils.utils import (
            _transform,
            get_capture_program_passes,
            tag_quant_io,
        )

        passes_job = get_capture_program_passes()
        passes_job[AnnotateDecomposed][QCOM_PASS_ACTIVATE_KEY] = True
        _transform(builder_exported_to_edge.edge_manager.exported_program(), passes_job)

        if config.misc.num_sharding > 0:
            model_sharding.split_graph(
                builder_exported_to_edge.edge_manager.exported_program(),
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=config.misc.num_sharding,
            )

        # pyre-ignore
        from executorch.backends.qualcomm.quantizer.custom_annotation import (
            get_custom_quant_ios_dtype,
        )

        atten = builder_exported_to_edge.model.layers[0].attention
        if config.backend.qnn.use_sha:
            cache_shape = torch.Size(
                (atten.max_batch_size, atten.max_context_len, atten.head_dim)
            )
        else:
            cache_shape = torch.Size(
                (
                    atten.max_batch_size,
                    atten.max_context_len,
                    atten.n_kv_heads,
                    atten.head_dim,
                )
            )
        tag_quant_io(
            builder_exported_to_edge.edge_manager.exported_program().graph_module,
            partial(get_custom_quant_ios_dtype, cache_shape),
        )

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    if config.misc.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if config.misc.verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        if config.misc.num_sharding > 0 and config.backend.qnn.enabled:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch(
            passes=additional_passes,
        )

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
        if config.misc.verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        if config.misc.num_sharding > 0 and config.backend.qnn.enabled:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch(passes=additional_passes)

    return builder


def _export_llama(config: DictConfig) -> LLMEdgeManager:  # noqa: C901
    _validate_config(config)

    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(config)

    additional_passes = []
    if config.model.name in TORCHTUNE_DEFINED_MODELS:
        additional_passes = [InitializedMutableBufferPass(["kv_cache_pos"])]

    # export_to_edge
    builder_exported = _prepare_for_llama_export(config).export()
    builder_exported.run_canonical_optimizations()
    modelname = builder_exported.modelname

    if config.export.export_only:
        exit()

    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        # Force xnnpack to be true if pt2e_quant_params is not None and config.backend.xnnpack.enabled is False
        config.backend.xnnpack.enabled = True

    if config.backend.xnnpack.enabled:
        builder = _to_edge_and_lower_llama_xnnpack(
            builder_exported,
            modelname,
            additional_passes,
            pt2e_quant_params,
            quantizers,
            quant_dtype,
            config,
        )
    else:
        builder = _to_edge_and_lower_llama(
            builder_exported,
            modelname,
            additional_passes,
            pt2e_quant_params,
            quantizers,
            quant_dtype,
            config,
        )

    if config.misc.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if config.export.output_name:
        modelname = config.export.output_name
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


def _validate_config(config: DictConfig) -> None:
    """Validate configuration values."""
    if config.sequence.max_context_length < config.sequence.max_seq_length:
        raise ValueError(
            f"max_context_length {config.sequence.max_context_length} must be >= max_seq_len {config.sequence.max_seq_length}. "
            "max_context_length impacts kv cache size that is used to remember history, while max_seq_length refers to user prompt length. "
            "Please use --max_context_length to specify context length."
        )

    if config.misc.enable_dynamic_shape and (
        config.backend.coreml.enabled
        or config.backend.mps.enabled
        or config.backend.qnn.enabled
    ):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends. "
            "Please use --disable_dynamic_shape."
        )

    if config.misc.num_sharding > 0 and not config.backend.qnn.enabled:
        raise ValueError("Model shard is only supported with qnn backend now.")

    if (
        config.quantization.mode is not None
        and config.quantization.mode.startswith("torchao:")
    ) or (
        config.quantization.embedding_quantize is not None
        and config.quantization.embedding_quantize.startswith("torchao:")
    ):
        if config.misc.enable_dynamic_shape:
            raise ValueError(
                "Dynamic shape is not currently supported with torchao ops. Please use --disable_dynamic_shape. "
                "If you need this feature, please file an issue."
            )


def _load_llama_model_metadata(
    weight_type: WeightType,
    use_kv_cache: bool,
    use_sdpa_with_kv_cache: bool,
    enable_dynamic_shape: bool,
    max_seq_len: int,
    max_context_len: int,
    n_layers: int,
    vocab_size: int,
    metadata_str: Optional[str] = None,
):
    is_fairseq2 = weight_type == WeightType.FAIRSEQ2
    metadata = {
        "get_bos_id": 3 if is_fairseq2 else 1,
        "get_eos_ids": [3] if is_fairseq2 else [2],
        "get_max_seq_len": max_seq_len,
        "get_max_context_len": max_context_len,
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
    params_path: Optional[str] = None,
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
    max_context_len: int = 128,
    input_prune_map_path: Optional[str] = None,
    output_prune_map_path: Optional[str] = None,
    metadata_str: Optional[str] = None,
    dtype_override: Optional[DType] = None,
    config: Optional[DictConfig] = None,
) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """

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

    torch_dtype = dtype_override.to_torch_dtype() if dtype_override else None

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
            max_context_len=max_context_len,
            enable_dynamic_shape=enable_dynamic_shape,
            input_prune_map_path=input_prune_map_path,
            output_prune_map_path=output_prune_map_path,
            dtype=torch_dtype,
            config=config,
        )
    )

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.max_seq_len,
        dtype=dtype_override,
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
            # pyre-fixme[6]: For 6th argument expected `ModelArgs` but got
            #  `Union[Tensor, Module]`.
            model.max_context_len,
            # pyre-fixme[6]: For 7th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.n_layers,
            # pyre-fixme[6]: For 8th argument expected `int` but got `Union[Tensor,
            #  Module]`.
            model.vocab_size,
            metadata_str,
        ),
        config=config,  # TODO: Rename builder args field to config.
    )


def _get_source_transforms(  # noqa
    modelname: str,
    dtype_override: DType,
    *,
    checkpoint_dtype: Optional[DType] = None,
    config: DictConfig,
) -> List[Callable[[torch.nn.Module], torch.nn.Module]]:
    """
    Return a list of functions that transform a graph.

    Args:
        modelname: The name of the model.
        dtype_override: The dtype to use for the model.
        checkpoint_dtype: The dtype of the checkpoint. At the moment, if this is specified,
            it means that you want to run quantize transformations on the weights represented
            in their original dtype, while the overall dtype of the model maybe something
            different. If not specified, defaults to dtype_override.
        args: The arguments passed to the script.

    Returns:
        A list of transformation functions.
    """

    if not checkpoint_dtype:
        checkpoint_dtype = dtype_override

    transforms = []

    if config.quantization.use_spin_quant:
        if config.quantization.use_spin_quant == "cuda":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_cuda_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_cuda_for_spin_quant)
        elif config.quantization.use_spin_quant == "native":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_native_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_native_for_spin_quant)

    if config.quantization.embedding_quantize:
        """
        When this option is selected, it finds all embedding layers and transforms
        into quantized embedding equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        this wil be a no-op.
        """
        modelname = f"{modelname}_e"
        transforms.append(get_quant_embedding_transform(config, checkpoint_dtype))

    # quantization_mode should be applied after embedding_quantize
    # to support shared_embedding
    if config.quantization.mode:
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
            get_quant_weight_transform(
                config=config,
                computation_dtype=dtype_override,
                checkpoint_dtype=checkpoint_dtype,
            )
        )

    if config.misc.expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    if config.kv_cache.use_sdpa_with_kv_cache:
        transforms.append(replace_kv_cache_with_custom_kv_cache)
        transforms.append(replace_sdpa_with_custom_op)

    if config.kv_cache.quantize_kv_cache:
        assert (
            config.kv_cache.use_kv_cache
        ), "quantize_kv_cache requires use_kv_cache=True"
        transforms.append(replace_kv_cache_with_quantized_kv_cache)

    if config.kv_cache.use_kv_cache:
        if config.backend.qnn.enabled:
            from executorch.backends.qualcomm.utils.utils import (
                convert_linear_to_conv2d,
            )

            if config.backend.qnn.use_sha:
                if config.backend.qnn.optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(
                        get_model_with_r1_r2(config.backend.qnn.optimized_rotation_path)
                    )
                transforms.append(replace_attention_to_attention_sha)
                transforms.append(replace_causal_mask)
                transforms.append(replace_rms_norm_with_native_rms_norm)
                # pyre-ignore: Module `backends` has no attribute `qualcomm`.
                transforms.append(convert_linear_to_conv2d)
            else:
                transforms.append(replace_kv_cache_with_simple_kv_cache)
                transforms.append(replace_sdpa_with_flex_sdpa)
                transforms.append(replace_causal_mask)
                transforms.append(replace_rms_norm_with_native_rms_norm)
                if config.backend.qnn.optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(
                        get_model_with_r1_r2(config.backend.qnn.optimized_rotation_path)
                    )
                # pyre-ignore: Module `backends` has no attribute `qualcomm`.
                transforms.append(convert_linear_to_conv2d)

        elif config.backend.mps.enabled:
            # Currently mps doesn't support sdpa op, use the simpler decomposition
            # to get free perf gain.
            transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_causal_mask)

        elif config.backend.coreml.enabled:
            # iOS 18 introduced fused sdpa op
            if config.backend.coreml.ios >= 18:
                transforms.append(replace_sdpa_with_coreml_sdpa)
            else:
                transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_kv_cache_with_coreml_kv_cache)

    if config.backend.vulkan.enabled:
        transforms.append(replace_with_vulkan_rotary_emb)

    return transforms


def get_llama_model(args):
    _validate_args(args)
    e_mgr = _prepare_for_llama_export(args)
    model = (
        e_mgr.model.eval().to(device="cuda")
        if torch.cuda.is_available()
        else e_mgr.model.eval().to(device="cpu")
    )
    return model, e_mgr.example_inputs, e_mgr.metadata
