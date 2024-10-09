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
import shlex
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, List, Optional, Union

import pkg_resources

import torch

from executorch.devtools.etrecord import generate_etrecord

from executorch.examples.models.llama2.llama_transformer import ModelArgs

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
)
from executorch.util.activation_memory_profiler import generate_memory_trace

from ..model_factory import EagerModelFactory
from .source_transformation.apply_spin_quant_r1_r2 import (
    fuse_layer_norms,
    get_model_with_r1_r2,
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
    replace_sdpa_with_simple_sdpa,
)

IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


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
    modelname: str = "model",
    extra_opts: str = "",
    *,
    par_local_output: bool = False,
    resource_pkg_name: str = __name__,
) -> str:
    if False:  # par_local_output:
        output_dir_path = "par:."
    else:
        output_dir_path = "."

    argString = f"--checkpoint par:{modelname}_ckpt.pt --params par:{modelname}_params.json {extra_opts} --output-dir {output_dir_path}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    # pkg_name = resource_pkg_name
    return export_llama(modelname, args)


def build_args_parser() -> argparse.ArgumentParser:
    ckpt_dir = f"{Path(__file__).absolute().parent.as_posix()}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    # parser.add_argument(
    #     "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    # )
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
        ],
        help="Use PT2E quantization. Comma separated options. e.g. xnnpack_dynamic (for per channel 8 bit weight), xnnpack_dynamic_qc4 (for per channel 4 bit weight), embedding.",
    )
    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=str,
        default=None,
        choices=["int8", "8da4w", "8da4w-gptq"],
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
    parser.add_argument("-X", "--xnnpack", action="store_true")
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
        choices=["b4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight)",
    )
    parser.add_argument(
        "--coreml-ios",
        type=int,
        default=15,
        choices=(15, 16, 17, 18),
        help="This option is only for coreml: The minimum iOS version to deploy",
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


def export_llama(modelname, args) -> str:
    if args.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(args.profile_path):
                builder = _export_llama(modelname, args)
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
        builder = _export_llama(modelname, args)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(modelname: str, args) -> LLMEdgeManager:
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
            modelname=modelname,
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
            metadata_str=args.metadata,
            args=args,
        )
        .set_output_dir(output_dir_path)
        .to_dtype(dtype_override)
        .source_transform(_get_source_transforms(modelname, dtype_override, args))
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
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


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


def _export_llama(modelname, args) -> LLMEdgeManager:  # noqa: C901
    _validate_args(args)
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    # export_to_edge
    builder_exported_to_edge = (
        _prepare_for_llama_export(modelname, args)
        .capture_pre_autograd_graph()
        .pt2e_quantize(quantizers)
        .export_to_edge()
    )

    modelname = builder_exported_to_edge.modelname

    # to_backend
    partitioners = []
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        partitioners.append(
            get_vulkan_partitioner(
                args.dtype_override,
                args.quantization_mode,
            )
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
        from executorch.backends.qualcomm.utils.utils import _transform

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`, Optional type has no attribute `exported_program`
        _transform(builder_exported_to_edge.edge_manager.exported_program())

        if args.num_sharding > 0:
            model_sharding.split_graph(
                builder_exported_to_edge.edge_manager.exported_program(),
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=args.num_sharding,
            )

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

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
    model_args: ModelArgs,
    metadata_str: Optional[str] = None,
):
    is_fairseq2 = weight_type == WeightType.FAIRSEQ2
    metadata = {
        "append_eos_to_prompt": is_fairseq2,  # For language llama, tell the runtime to always append EOS token(s) to prompt.
        "get_bos_id": 3 if is_fairseq2 else 1,
        "get_eos_ids": [3] if is_fairseq2 else [2],
        "get_max_seq_len": model_args.max_seq_len,
        "get_n_bos": 1,
        "get_n_eos": 2 if is_fairseq2 else 1,
        "get_n_layers": model_args.n_layers,
        "get_vocab_size": model_args.vocab_size,
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
    *,
    modelname: str = "llama2",
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
    metadata_str: Optional[str] = None,
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
    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        params=params_path,
        use_kv_cache=use_kv_cache,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        generate_full_logits=generate_full_logits,
        fairseq2=weight_type == WeightType.FAIRSEQ2,
        max_seq_len=max_seq_len,
        enable_dynamic_shape=enable_dynamic_shape,
        args=args,
    )
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
        max_seq_len=model.params.max_seq_len,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
        generate_full_logits=generate_full_logits,
        example_inputs=example_inputs,
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
            model.params,
            metadata_str,
        ),
        args=args,
    )


def _get_source_transforms(  # noqa
    modelname: str, dtype_override: Optional[DType], args
) -> List[Callable[[torch.nn.Module], torch.nn.Module]]:
    transforms = []
    if args.quantization_mode:
        modelname = f"{modelname}_q"
        if args.use_spin_quant is None:
            transforms.append(
                get_quant_weight_transform(args, dtype_override, verbose_export())
            )
        # For SpinQuant, the checkpoints are already quantized
        # aka the weights have corresponding scales value,
        # So that means, we don't need to apply quantization
        # transform. However, we will still need to apply
        # transformations that change the model structure to
        # match the checkpoint format.
        # transform_for_spinquant() will apply these transformations
        # later in model.py file.
        elif args.use_spin_quant == "cuda":
            from .source_transformation.spin_quant import (
                inject_fast_hadamard_transform_cuda_for_spin_quant,
            )

            transforms.append(inject_fast_hadamard_transform_cuda_for_spin_quant)
        elif args.use_spin_quant == "native":
            raise NotImplementedError("native SpinQuant is not implemented yet.")

    if args.embedding_quantize:
        modelname = f"{modelname}_e"
        transforms.append(get_quant_embedding_transform(args))

    if args.expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    if args.use_sdpa_with_kv_cache:
        transforms.append(replace_sdpa_with_custom_op)

    if args.use_kv_cache:
        if args.qnn:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
            from executorch.backends.qualcomm.utils.utils import (
                convert_linear_to_conv2d,
            )

            transforms.append(replace_kv_cache_with_simple_kv_cache)
            transforms.append(replace_sdpa_with_flex_sdpa)
            transforms.append(replace_causal_mask)
            transforms.append(replace_rms_norm_with_native_rms_norm)
            if args.optimized_rotation_path:
                transforms.append(fuse_layer_norms)
                transforms.append(get_model_with_r1_r2(args.optimized_rotation_path))
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

    return transforms
