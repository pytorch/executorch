# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import logging
import os
import shlex

from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import pkg_resources
import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)

from executorch.examples.models.llama2.llama_transformer import (
    KVCache,
    SDPA,
    Transformer,
)
from executorch.exir.backend.backend_details import CompileSpec

from executorch.sdk.etrecord import generate_etrecord
from executorch.util.activation_memory_profiler import generate_memory_trace
from sentencepiece import SentencePieceProcessor

from .builder import DType, LlamaEdgeManager, load_llama_model, WeightType
from .quant_lib import _get_pt2e_quantization_params, get_pt2e_quantizers

from .quantize import EmbeddingOnlyInt8QuantHandler, WeightOnlyInt8QuantHandler


IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


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


def materialze_broadcast_of_rope_freq_cis(
    module: torch.nn.Module,
):
    assert isinstance(module, Transformer)
    assert module.freqs_cos.dim() == 2
    dim0 = module.freqs_cos.size(0)
    dim1 = module.freqs_cos.size(1)
    assert (
        module.layers[0].attention.n_local_kv_heads
        == module.layers[0].attention.n_local_heads
    ), f"For rope freqs to be materialzed for broadcast q, k, v num heads must match. For q got {module.attention.n_kv_heads} for k got {module.attention.n_local_heads} and v got {module.attention.n_local_kv_heads}"
    num_heads = module.layers[0].attention.n_local_heads
    module.freqs_cos = module.freqs_cos.view(dim0, 1, dim1)
    module.freqs_cos = module.freqs_cos.expand(dim0, num_heads, dim1).contiguous()
    assert module.freqs_sin.dim() == 2
    assert dim0 == module.freqs_sin.size(
        0
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 0: {dim0} vs {module.freqs_sin.size(0)}"
    assert dim1 == module.freqs_sin.size(
        1
    ), f"sin and cos freq table sizes must match. Mismatch found at dim 1: {dim1} vs {module.freqs_sin.size(1)}"
    module.freqs_sin = module.freqs_sin.view(dim0, 1, dim1)
    module.freqs_sin = module.freqs_sin.expand(dim0, num_heads, dim1).contiguous()
    return module


class SDPACustom(torch.nn.Module):
    def __init__(
        self,
        kv_cache: KVCache,
        mask,
        dim: int,
    ):
        super().__init__()
        self.kv_cache = kv_cache
        self.mask = mask
        self.dim = dim

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz,
        seqlen,
    ):
        output = torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            self.kv_cache.k_cache,
            self.kv_cache.v_cache,
            input_pos[-1].item(),
            seqlen,
        )
        return output.view(bsz, seqlen, self.dim)


def _replace_sdpa_with_custom_op(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPACustom(child.kv_cache, child.mask, child.dim),
            )
        else:
            _replace_sdpa_with_custom_op(child)


def replace_sdpa_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:
    from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache  # noqa

    _replace_sdpa_with_custom_op(module)
    return module


def quantize(
    model: torch.nn.Module,
    qmode: str,
    activation_dtype: Optional[DType],
    checkpoint_path: Optional[Path] = None,
    # following arguments only available when setting int4 or gptq quantization.
    group_size: Optional[int] = 128,
    # following arguments are only used for GPTQ
    calibration_tasks: Optional[list] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    tokenizer_path: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Quantizes a model by converting all weights to int8.
    Args:
        model: A model to quantize.
        qmode: quantization mode, e.g. int8, 8da4w, 8da4w-gptq
    Returns:
        A quantized model.
    """
    if activation_dtype is not None:
        torch_dtype = activation_dtype.to_torch_dtype()
    else:
        torch_dtype = torch.float16

    assert checkpoint_path, "Need to specify a checkpoint"
    assert os.path.isfile(
        canonical_path(checkpoint_path)
    ), f"{checkpoint_path} does not exist"
    # if checkpoint_path is None:
    #     checkpoint_path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")

    if qmode == "int8":
        # Add quantization mode options here: group size, bit width, etc.
        return WeightOnlyInt8QuantHandler(model).quantized_model()
    elif qmode == "8da4w":
        # Check for required args
        if group_size is None:
            raise Exception("For 8da4w quantization, group size must be specified.")
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        model = Int8DynActInt4WeightQuantizer(
            precision=torch_dtype, groupsize=group_size
        ).quantize(model)
        if verbose_export():
            print("quantized model:", model)
        return model
    elif qmode == "8da4w-gptq":
        # Check for required args
        required_args: Optional[Any] = [
            group_size,
            calibration_limit,
            calibration_seq_length,
        ]
        if any(arg is None for arg in required_args):
            raise Exception(
                "For 8da4w-gptq quantization, group size, calibration limit and calibration sequence length must be specified."
            )
        if calibration_tasks is None:
            calibration_tasks = ["wikitext"]

        from torchao.quantization.GPTQ import InputRecorder
        from torchao.quantization.quant_api import Int8DynActInt4WeightGPTQQuantizer

        if tokenizer_path is None:
            tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
        )

        inputs = (
            InputRecorder(
                tokenizer,
                calibration_seq_length,
                None,  # input_prep_func
                pad_calibration_inputs,
                model.vocab_size,
            )
            .record_inputs(
                calibration_tasks,
                calibration_limit,
            )
            .get_inputs()
        )

        gptq_quantizer = Int8DynActInt4WeightGPTQQuantizer(
            blocksize,
            percdamp,
            group_size,
        )
        model = gptq_quantizer.quantize(model, inputs)
        return model
    else:
        raise Exception(f"Unrecognized quantize mode: {qmode}")


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
        help="Tasks for GPTQ calibration",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=None,
        help="Sequence length for GPTQ calibration",
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
        "--use_sdpa_with_kv_cache",
        default=False,
        action="store_true",
        help="Whether to use sdpa_with_kv_cache update op when using kv cache",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=f"{ckpt_dir}/params/demo_config.json",
        help="config.json",
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
        choices=["fp32"],
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp32",
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
                return _export_llama(modelname, args)
        except ImportError:
            print(
                "Please run `pip install snakeviz` to install required dependencies for cProfiler flamegraph."
            )
            return ""
    else:
        return _export_llama(modelname, args)


def _prepare_for_llama_export(modelname: str, args) -> LlamaEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LlamaEdgeManager with initial transforms and dtype conversion.

    Returns a LlamaEdgeManager prior to calling export_to_edge with quantizers
    """

    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint) if args.checkpoint else None
    checkpoint_dir = (
        canonical_path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    params_path = canonical_path(args.params)
    output_dir_path = canonical_path(args.output_dir, dir=True)
    modelname = "llama2"
    weight_type = WeightType.FAIRSEQ2 if args.fairseq2 else WeightType.LLAMA

    # dtype override
    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
    elif args.quantization_mode in ["8da4w", "8da4w-gptq"]:
        dtype_override = DType["fp16"]
    else:
        dtype_override = None

    # source transforms
    transforms = []
    if args.quantization_mode:
        modelname = f"{modelname}_q"

        # If these optional args are None, don't provide them to quantize()
        quant_args_str = [
            "group_size",
            "calibration_tasks",
            "calibration_limit",
            "calibration_seq_length",
        ]
        arg_dict = vars(args)
        quant_args = {
            param: val
            for param in quant_args_str
            if (val := arg_dict.get(param)) is not None
        }

        transforms.append(
            partial(
                quantize,
                **quant_args,
                qmode=args.quantization_mode,
                activation_dtype=dtype_override,
                checkpoint_path=(
                    Path(path) if (path := args.checkpoint) is not None else None
                ),
                tokenizer_path=(
                    Path(path) if (path := args.tokenizer_path) is not None else None
                ),
            )
        )

    if args.embedding_quantize:
        modelname = f"{modelname}_e"
        bitwidth, group_size = args.embedding_quantize.split(",")
        if group_size == "none" or group_size == "None" or group_size == "0":
            group_size = None
        else:
            group_size = int(group_size)
        bitwidth = int(bitwidth)
        transforms.append(
            lambda model: EmbeddingOnlyInt8QuantHandler(
                model, bitwidth=bitwidth, group_size=group_size
            ).quantized_model()
        )

    if args.expand_rope_table:
        transforms.append(materialze_broadcast_of_rope_freq_cis)

    if args.use_sdpa_with_kv_cache:
        transforms.append(replace_sdpa_with_custom_op)

    return (
        load_llama_model(
            checkpoint=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            params_path=params_path,
            use_kv_cache=args.use_kv_cache,
            use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
            weight_type=weight_type,
            verbose=args.verbose,
            max_seq_len=args.max_seq_length,
        )
        .set_output_dir(output_dir_path)
        .set_metadata(args.metadata)
        .source_transform(transforms)
        .to_dtype(dtype_override)
    )


def _export_llama(modelname, args) -> str:  # noqa: C901
    # export_to_edge
    pt2e_quant_params = _get_pt2e_quantization_params(args)
    quantizers = get_pt2e_quantizers(pt2e_quant_params, args)
    if args.qnn:
        assert (
            args.quantization_mode is None
        ), "Currently qnn backend only supports QnnQuantizer via pt2e flow"
        try:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.quantizer.quantizer`
            from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

            # reset quantizers and pt2e_quant_params from xnnpack backend
            pt2e_quant_params = None
            quantizers = []
        except ImportError:
            raise ImportError(
                "Please install the Qualcomm backend follwing https://pytorch.org/executorch/main/build-run-qualcomm.html"
            )

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`.
        qnn_quantizer = QnnQuantizer()
        # more custom quantization are supported including 16a4w etc. default to 8bit quantized
        custom_annotations = ()
        qnn_quantizer.add_custom_quant_annotations(custom_annotations)
        quantizers.append(qnn_quantizer)

    builder_exported_to_edge = _prepare_for_llama_export(
        modelname, args
    ).export_to_edge(quantizers)

    # to_backend
    partitioners = []
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioners.append(XnnpackDynamicallyQuantizedPartitioner())
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        # Following changes due to.
        # 1. We need dynamically quantized partitioner for both pt2e_quantize options
        #    as well as "qmode 8da4w" which is also dynamic quantizes linear layers.
        # 2. XNNPACK partitioner seems to result in seg fault for non dqlinear ops.
        partitioners.append(XnnpackDynamicallyQuantizedPartitioner())
        # partitioners.append(XnnpackPartitioner())
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        assert (
            args.dtype_override == "fp32" or args.dtype_override is None
        ), "Vulkan backend does not support non fp32 dtypes at the moment"
        assert (
            args.quantization_mode is None
        ), "Vulkan backend does not support quantization at the moment"

        partitioners.append(VulkanPartitioner())
        modelname = f"vulkan_{modelname}"

    if args.mps:
        assert (
            args.use_kv_cache is True
        ), "MPS backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.mps.partition.mps_partitioner`.
            from executorch.backends.apple.mps.partition.mps_partitioner import (
                MPSPartitioner,
            )
        except ImportError:
            raise ImportError(
                "Please install the MPS backend follwing https://pytorch.org/executorch/main/build-run-mps.html"
            )

        compile_specs = [CompileSpec("use_fp16", bytes([True]))]
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`.
        partitioners.append(MPSPartitioner(compile_specs))
        modelname = f"mps_{modelname}"

    if args.coreml:
        assert (
            args.use_kv_cache is True
        ), "CoreML backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.partition.coreml_partitioner`.
            import coremltools as ct

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.compiler`
            from executorch.backends.apple.coreml.compiler import CoreMLBackend

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.partition.coreml_partitioner`
            from executorch.backends.apple.coreml.partition.coreml_partitioner import (
                CoreMLPartitioner,
            )
        except ImportError:
            raise ImportError(
                "Please install the CoreML backend follwing https://pytorch.org/executorch/main/build-run-coreml.html"
            )

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`.
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision(ct.precision.FLOAT16.value),
            compute_unit=ct.ComputeUnit[ct.ComputeUnit.ALL.name.upper()],
            # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`
            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
        )
        partitioners.append(
            # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`
            CoreMLPartitioner(
                compile_specs=compile_specs,
            )
        )
        modelname = f"coreml_{modelname}"

    if args.qnn:
        assert (
            args.use_kv_cache is True
        ), "Qualcomm backend currently only supports static shape and use_kv_cache=True is the only way to support it at the moment"
        try:
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.partition.qnn_partitioner`
            from executorch.backends.qualcomm.partition.qnn_partitioner import (
                QnnPartitioner,
            )

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.serialization.qnn_compile_spec_schema`
            from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
                QcomChipset,
            )

            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
            from executorch.backends.qualcomm.utils.utils import (
                _transform,
                generate_htp_compiler_spec,
                generate_qnn_executorch_compiler_spec,
            )
        except ImportError:
            raise ImportError(
                "Please install the Qualcomm backend follwing https://pytorch.org/executorch/main/build-run-qualcomm.html"
            )

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        partitioners.append(
            # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
            QnnPartitioner(
                # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
                generate_qnn_executorch_compiler_spec(
                    # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`.
                    soc_model=QcomChipset.SM8650,  # default to SM8650
                    backend_options=backend_options,
                    debug=False,
                    saver=False,
                ),
                skip_node_id_set={},
                skip_node_op_set={},
            )
        )
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`
        _transform(builder_exported_to_edge.export_program())

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioners).to_executorch()

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord(
                etrecord_path="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        builder = builder_exported_to_edge.to_backend(partitioners).to_executorch()

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

    return output_file
