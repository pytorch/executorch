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

from executorch.devtools.backend_debug import print_delegation_info

from executorch.devtools.etrecord import generate_etrecord as generate_etrecord_func
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.config.llm_config import LlmConfig

from executorch.extension.llm.export.partitioner_lib import (
    get_coreml_partitioner,
    get_mps_partitioner,
    get_openvino_partitioner,
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

from omegaconf import DictConfig

from ..model_factory import EagerModelFactory
from .source_transformation.apply_spin_quant_r1_r2 import (
    fuse_layer_norms,
    get_model_with_r1_r2,
)

from .source_transformation.attention import replace_attention_to_attention_sha
from .source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
    replace_kv_cache_with_quantized_kv_cache,
    replace_kv_cache_with_ring_kv_cache,
)

from .source_transformation.quantize import (
    get_quant_embedding_transform,
    get_quant_weight_transform,
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
    replace_sdpa_with_quantized_sdpa,
    replace_sdpa_with_simple_sdpa,
)

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
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
    "phi_4_mini",
    "smollm2",
]
TORCHTUNE_DEFINED_MODELS = ["llama3_2_vision"]
HUGGING_FACE_REPO_IDS = {
    "qwen2_5": "Qwen/Qwen2.5-1.5B",
    "phi_4_mini": "microsoft/Phi-4-mini-instruct",
    "smollm2": "HuggingFaceTB/SmolLM-135M",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
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
    llm_config = LlmConfig.from_args(args)
    return export_llama(llm_config)


def parse_list_of_ints(s):
    import ast

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
            print(parsed)
            return parsed
        raise argparse.ArgumentTypeError(
            "Must be a list of integers, e.g., [0, 16, 0, 16]"
        )
    except Exception:
        raise argparse.ArgumentTypeError(
            "Must be a list of integers, e.g., [0, 16, 0, 16]"
        )


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
        "--adapter_checkpoint",
        required=False,
        help="Path to the adapter.pt file from torchtune. Used if the model has trained LoRA adapters. Must provide adapter_config.json",
    )

    parser.add_argument(
        "--adapter_config",
        required=False,
        help="Path to the adapter_config.json file. Used if the model has trained LoRA adapters. Must provide adapter_checkpoint.",
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

    parser.add_argument(
        "--local_global_attention",
        type=parse_list_of_ints,
        default=None,
        help="List of integers specifying local and global attention pattern, e.g., [0, 16, 0, 16] to specify that every other layer is sliding window of 16."
        " [0, 16, 32] pattern specifes 2nd and 3rd layer has sliding window of 16 and 32 respecitvely. "
        " [16] pattern specifies all layers have sliding window of 16.",
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
    parser.add_argument("--openvino", action="store_true")
    parser.add_argument(
        "--openvino_device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU"],
        help="Specify the device for Openvino (CPU or GPU).",
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
        help="Whether the checkpoint is pre-quantized with QAT or not.",
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
        "--nncf_compression",
        default=False,
        action="store_true",
        help="Enables nncf compression for openvino backend",
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


def export_llama(
    export_options: Union[argparse.Namespace, LlmConfig, DictConfig],
) -> str:
    if isinstance(export_options, argparse.Namespace):
        # Legacy CLI.
        llm_config = LlmConfig.from_args(export_options)
    elif isinstance(export_options, LlmConfig) or isinstance(
        export_options, DictConfig
    ):
        # Hydra CLI.
        llm_config = export_options
    else:
        raise ValueError(
            "Input to export_llama must be either of type argparse.Namespace or LlmConfig"
        )

    # If a checkpoint isn't provided for an HF OSS model, download and convert the
    # weights first.
    model_name = llm_config.base.model_class.value
    if not llm_config.base.checkpoint and model_name in HUGGING_FACE_REPO_IDS:
        repo_id = HUGGING_FACE_REPO_IDS[model_name]
        if model_name == "qwen2_5":
            from executorch.examples.models.qwen2_5 import (  # pyre-ignore[21]
                convert_weights,
            )
        elif model_name.startswith("qwen3"):
            from executorch.examples.models.qwen3 import (  # pyre-ignore[21]
                convert_weights,
            )
        elif model_name == "phi_4_mini":
            from executorch.examples.models.phi_4_mini import (  # pyre-ignore[21]
                convert_weights,
            )
        elif model_name == "smollm2":
            from executorch.examples.models.smollm2 import (  # pyre-ignore[21]
                convert_weights,
            )
        else:
            raise ValueError(
                f"Converting weights to meta format for {model_name} is not yet supported"
            )
        checkpoint = download_and_convert_hf_checkpoint(repo_id, convert_weights)
        llm_config.base.checkpoint = checkpoint

    if llm_config.debug.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(llm_config.debug.profile_path):
                builder = _export_llama(llm_config)
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
        builder = _export_llama(llm_config)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(llm_config: LlmConfig) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """
    # load model from checkpoint and params.json
    checkpoint_path = (
        canonical_path(llm_config.base.checkpoint)
        if llm_config.base.checkpoint
        else None
    )
    checkpoint_dir = (
        canonical_path(llm_config.base.checkpoint_dir)
        if llm_config.base.checkpoint_dir
        else None
    )
    params_path = (
        canonical_path(llm_config.base.params) if llm_config.base.params else None
    )
    output_dir_path = canonical_path(llm_config.export.output_dir, dir=True)

    llm_config.base.checkpoint = checkpoint_path
    llm_config.base.checkpoint_dir = checkpoint_dir
    llm_config.base.params = params_path
    llm_config.export.output_dir = output_dir_path

    # Convert dtype override string to actual type.
    dtype_override = DType[llm_config.model.dtype_override.value]

    edge_manager = _load_llama_model(llm_config)

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
            dtype_override=dtype_override,
            checkpoint=llm_config.base.checkpoint,
            checkpoint_dtype=DType.from_torch_dtype(checkpoint_dtype),  # type: ignore
            tokenizer_path=llm_config.base.tokenizer_path,
            use_spin_quant=(
                llm_config.quantization.use_spin_quant.value
                if llm_config.quantization.use_spin_quant
                else None
            ),
            embedding_quantize=llm_config.quantization.embedding_quantize,
            use_shared_embedding=llm_config.model.use_shared_embedding,
            quantization_mode=llm_config.quantization.qmode,
            group_size=llm_config.quantization.group_size,
            calibration_tasks=llm_config.quantization.calibration_tasks,
            calibration_limit=llm_config.quantization.calibration_limit,
            calibration_seq_length=llm_config.quantization.calibration_seq_length,
            expand_rope_table=llm_config.model.expand_rope_table,
            use_custom_sdpa_with_attention_mask=getattr(
                llm_config.model, "use_custom_sdpa_with_attention_mask", False
            ),
            use_sdpa_with_kv_cache=llm_config.model.use_sdpa_with_kv_cache,
            quantize_kv_cache=llm_config.model.quantize_kv_cache,
            use_kv_cache=llm_config.model.use_kv_cache,
            qnn=llm_config.backend.qnn.enabled,
            use_qnn_sha=llm_config.backend.qnn.use_sha,
            optimized_rotation_path=llm_config.backend.qnn.optimized_rotation_path,
            mps=llm_config.backend.mps.enabled,
            coreml=llm_config.backend.coreml.enabled,
            coreml_ios=llm_config.backend.coreml.ios,
            vulkan=llm_config.backend.vulkan.enabled,
            use_qat=llm_config.quantization.use_qat,
            use_lora=llm_config.base.use_lora,
            preq_mode=(
                llm_config.base.preq_mode.value if llm_config.base.preq_mode else None
            ),
            preq_group_size=llm_config.base.preq_group_size,
            preq_embedding_quantize=llm_config.base.preq_embedding_quantize,
            local_global_attention=llm_config.model.local_global_attention,
        )
    )

    return edge_manager


def get_quantizer_and_quant_params(llm_config):
    pt2e_quant_params = get_pt2e_quantization_params(
        (
            llm_config.quantization.pt2e_quantize.value
            if llm_config.quantization.pt2e_quantize
            else None
        ),
        llm_config.quantization.qmode,
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, llm_config.export.so_library)
    quant_dtype = None
    if llm_config.backend.qnn.enabled and llm_config.quantization.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack and qnn"
        qnn_quantizer, quant_dtype = get_qnn_quantizer(
            llm_config.quantization.pt2e_quantize.value, llm_config.quantization.qmode
        )
        quantizers.append(qnn_quantizer)
    if llm_config.backend.coreml.enabled and llm_config.quantization.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack / qnn and coreml"
        coreml_quantizer = get_coreml_quantizer(
            llm_config.quantization.pt2e_quantize.value
        )
        quantizers.append(coreml_quantizer)
    if llm_config.backend.vulkan.enabled and llm_config.quantization.pt2e_quantize:
        assert (
            len(quantizers) == 0
        ), "Should not enable both vulkan and other quantizers"
        vulkan_quantizer = get_vulkan_quantizer(
            llm_config.quantization.pt2e_quantize.value
        )
        quantizers.append(vulkan_quantizer)
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


def _qmode_type(value):
    choices = ["int8", "8da4w", "8da4w-gptq", "vulkan_4w", "4w"]
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


def _validate_args(llm_config):
    if llm_config.export.max_context_length < llm_config.export.max_seq_length:
        raise ValueError(
            f"max_context_length {llm_config.export.max_context_length} must be >= max_seq_len {llm_config.export.max_seq_length}. max_context_length impacts kv cache size that is used to remember history, while max_seq_length refers to user prompt length. Please use --max_context_length to specify context length."
        )
    if llm_config.model.enable_dynamic_shape and (
        llm_config.backend.coreml.enabled
        or llm_config.backend.mps.enabled
        or llm_config.backend.qnn.enabled
    ):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends."
            " Please use --disable_dynamic_shape."
        )

    if llm_config.backend.qnn.num_sharding > 0 and not llm_config.backend.qnn.enabled:
        raise ValueError("Model shard is only supported with qnn backend now.")

    if llm_config.model.use_shared_embedding:
        if not (
            llm_config.quantization.embedding_quantize is not None
            and llm_config.quantization.embedding_quantize.startswith("torchao:")
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
    xnnpack_extended_ops: bool = False,
    generate_etrecord: bool = False,
    verbose: bool = False,
) -> LLMEdgeManager:  # noqa: C901
    partitioners = []

    # Order matters here, dynamic quantization should be applied first when both xnnpack and xnnpack_extended_ops are enabled
    partitioners.append(get_xnnpack_partitioner(dynamic_quant_only_partitioner=True))

    modelname = f"xnnpack_dq_{modelname}"

    if xnnpack_extended_ops:
        partitioners.append(
            get_xnnpack_partitioner(dynamic_quant_only_partitioner=False)
        )
        modelname = f"xnnpack_{modelname}"

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    # TODO: Enable generating ETRecord with XNNPack and to_edge_transform_and_lower().
    if generate_etrecord:
        raise NotImplementedError(
            "export_llama does not support XNNPack and generating ETRecord at the moment."
        )

    builder = builder_exported.pt2e_quantize(quantizers).to_edge_transform_and_lower(
        partitioners
    )
    if verbose:
        print_delegation_info(builder.edge_manager.exported_program().graph_module)

    return builder.to_executorch(passes=additional_passes)


def _to_edge_and_lower_llama_openvino(
    builder_exported,
    modelname,
    additional_passes,
    openvino_device: str = "CPU",
    nncf_compression: bool = False,
    nncf_compression_group_size: int = 32,
    verbose: bool = False,
) -> LLMEdgeManager:  # noqa: C901
    partitioners = []

    # Add OpenVINO partitioner
    partitioners.append(get_openvino_partitioner(openvino_device))
    modelname = f"openvino_{modelname}"

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    # Use NNCF compression if enabled
    # TODO: Enable passing OpenVINOQuantizer as a parameter to pt2e_quantize
    if nncf_compression:
        try:
            from functools import partial

            import nncf
            from pytorch_tokenizers import get_tokenizer
        except ImportError:
            raise ImportError(
                "Please install nncf via backends/openvino/requirements.txt"
            )
        tokenizer = get_tokenizer(builder_exported.tokenizer_path)

        def transform_fn(prompts: str, tokenizer):
            tokenized_text = tokenizer.encode(prompts, bos=False, eos=False)
            logging.error(tokenized_text)

            inputs = ()
            inputs = (
                torch.tensor(tokenized_text).unsqueeze(0),
                {"input_pos": torch.tensor([0])},
            )

            return inputs

        builder_exported.calibration_data = (
            [builder_exported.calibration_data]
            if isinstance(builder_exported.calibration_data, str)
            else builder_exported.calibration_data
        )
        builder_exported.calibration_data = (
            [
                word
                for prompt in builder_exported.calibration_data
                for word in prompt.split()
            ]
            if not builder_exported.dynamic_shapes
            else builder_exported.calibration_data
        )

        builder_exported.pre_autograd_graph_module = nncf.compress_weights(
            builder_exported.pre_autograd_graph_module,
            dataset=nncf.Dataset(
                builder_exported.calibration_data,
                transform_func=partial(transform_fn, tokenizer=tokenizer),
            ),
            mode=nncf.CompressWeightsMode.INT4_SYM,
            ratio=0.8,
            group_size=nncf_compression_group_size,
            sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        )

    builder = builder_exported.to_edge_transform_and_lower(partitioners)

    if verbose:
        print_delegation_info(builder.edge_manager.exported_program().graph_module)

    return builder.to_executorch(passes=additional_passes)


def _to_edge_and_lower_llama(  # noqa: C901
    builder_exported,
    modelname,
    additional_passes,
    pt2e_quant_params,
    quantizers,
    quant_dtype,
    vulkan: bool = False,
    mps: bool = False,
    coreml: bool = False,
    qnn: bool = False,
    dtype_override: str = "fp32",
    enable_dynamic_shape: bool = True,
    use_kv_cache: bool = False,
    embedding_quantize: Optional[str] = None,
    pt2e_quantize: Optional[str] = None,
    coreml_ios: int = 15,
    coreml_quantize: Optional[str] = None,
    coreml_compute_units: str = "cpu_only",
    use_qnn_sha: bool = False,
    num_sharding: int = 0,
    soc_model: str = "SM8650",
    generate_etrecord: bool = False,
    verbose: bool = False,
):
    builder_exported_to_edge = builder_exported.pt2e_quantize(
        quantizers
    ).export_to_edge()

    # to_backend
    partitioners = []
    if vulkan:
        partitioners.append(
            get_vulkan_partitioner(
                dtype_override,
                enable_dynamic_shape,
            )
        )
        modelname = f"vulkan_{modelname}"

    if mps:
        partitioners.append(get_mps_partitioner(use_kv_cache))
        modelname = f"mps_{modelname}"

    if coreml:
        coreml_partitioner = get_coreml_partitioner(
            coreml_ios,
            embedding_quantize,
            pt2e_quantize,
            coreml_quantize,
            coreml_compute_units,
        )
        partitioners.append(coreml_partitioner)
        modelname = f"coreml_{modelname}"

    if qnn:
        logging.warning(
            "The model definition in current repro is not performant, please refer to the instruction"
            " in https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama/README.md for better performance."
        )
        from executorch.extension.llm.custom_ops import model_sharding

        partitioners.append(
            get_qnn_partitioner(use_kv_cache, pt2e_quantize, num_sharding, soc_model)
        )
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm._passes`
        from executorch.backends.qualcomm._passes import (
            AnnotateStack,
            ConvertBmmToMatmul,
            FoldQDQ,
            RecomposeRmsNorm,
            TagQuantIO,
        )

        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm._passes.qnn_pass_manager`
        from executorch.backends.qualcomm._passes.qnn_pass_manager import (
            get_capture_program_passes,
            get_passes_dependency_for_capture_program,
            QnnPassManager,
        )

        # pyre-ignore
        from executorch.backends.qualcomm.quantizer.custom_annotation import (
            get_custom_quant_ios_dtype,
        )

        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.constants`
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
        )

        atten = builder_exported_to_edge.model.layers[0].attention
        if use_qnn_sha:
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

        # TODO: Use to_edge_lower_and_transform for QNN
        passes_job = get_capture_program_passes()
        dep_table = get_passes_dependency_for_capture_program()
        passes_job[AnnotateStack][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[ConvertBmmToMatmul][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[RecomposeRmsNorm][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(get_custom_quant_ios_dtype, cache_shape)
        if num_sharding > 0:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=num_sharding,
            )
            passes_job[SplitGraph] = setting
            dep_table[SplitGraph] = [FoldQDQ]
            dep_table[TagQuantIO] = [SplitGraph]
        QnnPassManager().transform_for_to_edge_pipeline(
            builder_exported_to_edge.edge_manager.exported_program(),
            dep_table=dep_table,
            passes_job=passes_job,
        )

    logging.info("Lowering model using following partitioner(s): ")
    for partitioner in partitioners:
        logging.info(f"--> {partitioner.__class__.__name__}")

    if generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        if num_sharding > 0 and qnn:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`.
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch(
            passes=additional_passes,
        )

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord_func(
                et_record="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        builder = builder_exported_to_edge.to_backend(partitioners)
        if verbose:
            print_delegation_info(builder.edge_manager.exported_program().graph_module)
        if num_sharding > 0 and qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch(passes=additional_passes)

    return builder


def _export_llama(llm_config: LlmConfig) -> LLMEdgeManager:  # noqa: C901
    _validate_args(llm_config)

    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(
        llm_config
    )

    additional_passes = []
    if llm_config.base.model_class.value in TORCHTUNE_DEFINED_MODELS:
        additional_passes = [InitializedMutableBufferPass(["kv_cache_pos"])]

    # export_to_edge
    builder_exported = _prepare_for_llama_export(llm_config).export()
    builder_exported.run_canonical_optimizations()
    modelname = builder_exported.modelname

    if llm_config.export.export_only:
        exit()

    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        # Force xnnpack to be true if pt2e_quant_params is not None and xnnpack is False
        llm_config.backend.xnnpack.enabled = True

    if llm_config.backend.xnnpack.enabled:
        if llm_config.export.foundation_weights_file is not None:
            gen_tag_fn: Callable[[torch.fx.Node], Optional[str]] = lambda x: (
                llm_config.export.foundation_weights_file
                if "lora" not in x.name
                else None
            )

            from executorch.exir.passes.external_constants_pass import (
                delegate_external_constants_pass_unlifted,
            )

            assert (
                builder_exported.pre_autograd_graph_module is not None
            ), "pre_autograd_graph_module shouldn't be None here"
            delegate_external_constants_pass_unlifted(
                module=builder_exported.pre_autograd_graph_module,
                gen_tag_fn=gen_tag_fn,
            )

        builder = _to_edge_and_lower_llama_xnnpack(
            builder_exported,
            modelname,
            additional_passes,
            pt2e_quant_params,
            quantizers,
            quant_dtype,
            xnnpack_extended_ops=llm_config.backend.xnnpack.extended_ops,
            generate_etrecord=llm_config.debug.generate_etrecord,
            verbose=llm_config.debug.verbose,
        )
    elif llm_config.backend.openvino.enabled:
        builder = _to_edge_and_lower_llama_openvino(
            builder_exported,
            modelname,
            additional_passes,
            openvino_device=llm_config.backend.openvino.device,
            nncf_compression=llm_config.backend.openvino.nncf_compression,
            nncf_compression_group_size=llm_config.backend.openvino.nncf_compression_group_size,
            verbose=llm_config.debug.verbose,
        )
    else:
        builder = _to_edge_and_lower_llama(
            builder_exported,
            modelname,
            additional_passes,
            pt2e_quant_params,
            quantizers,
            quant_dtype,
            vulkan=llm_config.backend.vulkan.enabled,
            mps=llm_config.backend.mps.enabled,
            coreml=llm_config.backend.coreml.enabled,
            qnn=llm_config.backend.qnn.enabled,
            dtype_override=llm_config.model.dtype_override.value,
            enable_dynamic_shape=llm_config.model.enable_dynamic_shape,
            use_kv_cache=llm_config.model.use_kv_cache,
            embedding_quantize=llm_config.quantization.embedding_quantize,
            pt2e_quantize=(
                llm_config.quantization.pt2e_quantize.value
                if llm_config.quantization.pt2e_quantize
                else None
            ),
            coreml_ios=llm_config.backend.coreml.ios,
            coreml_quantize=(
                llm_config.backend.coreml.quantize.value
                if llm_config.backend.coreml.quantize
                else None
            ),
            coreml_compute_units=llm_config.backend.coreml.compute_units.value,
            use_qnn_sha=llm_config.backend.qnn.use_sha,
            num_sharding=llm_config.backend.qnn.num_sharding,
            soc_model=llm_config.backend.qnn.soc_model,
            generate_etrecord=llm_config.debug.generate_etrecord,
            verbose=llm_config.debug.verbose,
        )

    if llm_config.debug.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if llm_config.export.output_name:
        modelname = llm_config.export.output_name
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


def _load_llama_model(llm_config: LlmConfig) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """

    modelname = llm_config.base.model_class.value
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
            llm_config=llm_config,
        )
    )
    # Convert dtype override string to actual type.
    dtype_override = DType[llm_config.model.dtype_override.value]

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.max_seq_len,  # type: ignore
        dtype=dtype_override,
        use_kv_cache=llm_config.model.use_kv_cache,
        generate_full_logits=llm_config.debug.generate_full_logits,
        example_inputs=example_inputs,
        example_kwarg_inputs=example_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        enable_dynamic_shape=llm_config.model.enable_dynamic_shape,
        calibration_tasks=llm_config.quantization.calibration_tasks,
        calibration_limit=llm_config.quantization.calibration_limit,
        calibration_seq_length=llm_config.quantization.calibration_seq_length,
        calibration_data=llm_config.quantization.calibration_data,
        tokenizer_path=llm_config.base.tokenizer_path,
        save_exported_program=llm_config.export.export_only,
        verbose=llm_config.debug.verbose,
        metadata=_load_llama_model_metadata(
            WeightType.FAIRSEQ2 if llm_config.base.fairseq2 else WeightType.LLAMA,
            llm_config.model.use_kv_cache,
            llm_config.model.use_sdpa_with_kv_cache,
            llm_config.model.enable_dynamic_shape,
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
            llm_config.base.metadata,
        ),
    )


def _get_source_transforms(  # noqa
    dtype_override: DType,
    *,
    checkpoint: Optional[str] = None,
    checkpoint_dtype: Optional[DType] = None,
    tokenizer_path: Optional[str] = None,
    use_spin_quant: Optional[str] = None,
    embedding_quantize: Optional[str] = None,
    use_shared_embedding: bool = False,
    quantization_mode: Optional[str] = None,
    group_size: Optional[int] = None,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    expand_rope_table: bool = False,
    use_custom_sdpa_with_attention_mask: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    quantize_kv_cache: bool = False,
    use_kv_cache: bool = False,
    qnn: bool = False,
    use_qnn_sha: bool = False,
    optimized_rotation_path: Optional[str] = None,
    mps: bool = False,
    coreml: bool = False,
    coreml_ios: int = 15,
    vulkan: bool = False,
    use_qat: bool = False,
    use_lora: int = 0,
    preq_mode: Optional[str] = None,
    preq_group_size: Optional[int] = None,
    preq_embedding_quantize: Optional[str] = None,
    local_global_attention: Optional[List[int]] = None,
) -> List[Callable[[torch.nn.Module], torch.nn.Module]]:
    """
    Return a list of functions that transform a graph.

    Args:
        dtype_override: The dtype to use for the model.
        checkpoint: Path to the checkpoint file.
        checkpoint_dtype: The dtype of the checkpoint. At the moment, if this is specified,
            it means that you want to run quantize transformations on the weights represented
            in their original dtype, while the overall dtype of the model maybe something
            different. If not specified, defaults to dtype_override.
        tokenizer_path: Path to the tokenizer file.
        use_spin_quant: Type of spin quant to use ("cuda" or "native").
        embedding_quantize: Type of embedding quantization.
        quantization_mode: Type of quantization mode.
        expand_rope_table: Whether to expand rope table.
        use_custom_sdpa_with_attention_mask: Whether to use custom SDPA with attention mask.
        use_sdpa_with_kv_cache: Whether to use SDPA with KV cache.
        quantize_kv_cache: Whether to quantize KV cache.
        use_kv_cache: Whether to use KV cache.
        qnn: Whether to use QNN.
        use_qnn_sha: Whether to use QNN SHA.
        optimized_rotation_path: Path to optimized rotation.
        mps: Whether to use MPS.
        coreml: Whether to use CoreML.
        coreml_ios: CoreML iOS version.
        vulkan: Whether to use Vulkan.
        use_shared_embedding: Whether to use shared embedding.
        use_qat: Whether to use QAT.
        use_lora: LoRA rank (0 means no LoRA).
        preq_mode: Pre-quantization mode.
        preq_group_size: Pre-quantization group size.
        preq_embedding_quantize: Pre-quantization embedding quantize.

    Returns:
        A list of transformation functions.
    """

    if not checkpoint_dtype:
        checkpoint_dtype = dtype_override

    transforms = []

    if use_spin_quant:
        if use_spin_quant == "cuda":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_cuda_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_cuda_for_spin_quant)
        elif use_spin_quant == "native":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_native_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_native_for_spin_quant)

    if embedding_quantize:
        """
        When this option is selected, it finds all embedding layers and transforms
        into quantized embedding equivalent module.

        There are cases where the checkpoint is already quantized, for example
        on use_spin_quant is enabled. In that case, it will do the appropriate
        transformations based on the given checkpoint first. In those cases,
        this wil be a no-op.
        """
        transforms.append(
            get_quant_embedding_transform(
                embedding_quantize, use_shared_embedding, checkpoint_dtype
            )
        )

    # quantization_mode should be applied after embedding_quantize
    # to support shared_embedding
    if quantization_mode:
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
        transforms.append(
            get_quant_weight_transform(
                quantization_mode=quantization_mode,
                group_size=group_size,
                computation_dtype=dtype_override,
                checkpoint_dtype=checkpoint_dtype,
                checkpoint_path=checkpoint,
                tokenizer_path=tokenizer_path,
                calibration_tasks=calibration_tasks,
                calibration_limit=calibration_limit,
                calibration_seq_length=calibration_seq_length,
            )
        )

    if expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    use_attention_mask_for_custom_sdpa = use_custom_sdpa_with_attention_mask

    if use_sdpa_with_kv_cache:
        transforms.append(replace_kv_cache_with_custom_kv_cache)
        # todo: do this optionally
        # if use attention mask instead of causal attention
        # then create partial function that sets use_attention_mask=True
        if use_attention_mask_for_custom_sdpa:
            transforms.append(
                partial(replace_sdpa_with_custom_op, use_attention_mask=True)
            )
        else:
            transforms.append(replace_sdpa_with_custom_op)

    if quantize_kv_cache:
        assert use_kv_cache, "quantize_kv_cache requires use_kv_cache=True"
        transforms.append(replace_kv_cache_with_quantized_kv_cache)
        # Right now
        transforms.append(replace_sdpa_with_quantized_sdpa)

    if use_kv_cache:
        if qnn:
            from executorch.backends.qualcomm.utils.utils import (
                convert_linear_to_conv2d,
            )

            if use_qnn_sha:
                if optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(get_model_with_r1_r2(optimized_rotation_path))
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
                if optimized_rotation_path:
                    transforms.append(fuse_layer_norms)
                    transforms.append(get_model_with_r1_r2(optimized_rotation_path))
                # pyre-fixme[16]: Module `backends` has no attribute `qualcomm`.
                transforms.append(convert_linear_to_conv2d)

        elif mps:
            # Currently mps doesn't support sdpa op, use the simpler decomposition
            # to get free perf gain.
            transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_causal_mask)

        elif coreml:
            # iOS 18 introduced fused sdpa op
            if coreml_ios >= 18:
                transforms.append(replace_sdpa_with_coreml_sdpa)
            else:
                transforms.append(replace_sdpa_with_simple_sdpa)
            transforms.append(replace_kv_cache_with_coreml_kv_cache)

    if local_global_attention:
        transforms.append(
            partial(
                replace_kv_cache_with_ring_kv_cache,
                layer_sizes=local_global_attention,
            )
        )

    return transforms


def get_llama_model(llm_config: LlmConfig):
    _validate_args(llm_config)
    e_mgr = _prepare_for_llama_export(llm_config)
    model = (
        e_mgr.model.eval().to(device="cuda")
        if torch.cuda.is_available()
        else e_mgr.model.eval().to(device="cpu")
    )
    return model, e_mgr.example_inputs, e_mgr.metadata
