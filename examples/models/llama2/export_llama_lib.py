# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import logging
import shlex
from dataclasses import dataclass

from functools import partial
from pathlib import Path
from typing import List, Optional

import pkg_resources
import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)

from executorch.examples.models.llama2.llama_transformer import Transformer
from executorch.exir.backend.backend_details import CompileSpec

from executorch.sdk.etrecord import generate_etrecord
from executorch.util.activation_memory_profiler import generate_memory_trace
from sentencepiece import SentencePieceProcessor
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.embedding_quantizer import EmbeddingQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from .builder import DType, LlamaEdgeManager, load_llama_model, WeightType

from .quantize import EmbeddingOnlyInt8QuantHandler, WeightOnlyInt8QuantHandler


IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__


def set_pkg_name(name: str) -> None:
    global pkg_name
    pkg_name = name


def get_resource_path(resource_name) -> str:
    return pkg_resources.resource_filename(pkg_name, resource_name)


@dataclass
class EmbeddingQuantOptions:
    is_per_channel: bool = True
    group_size: int = -1

    def __post_init__(self):
        if self.group_size != -1:
            raise RuntimeError(
                "PT2E embedding quantizer does not support groupwise at the moment."
            )


@dataclass
class DynamicQuantLinearOptions:
    is_per_channel: bool = True
    is_qc4: bool = False


@dataclass
class PT2EQuantOptions:
    quantize_embedding: Optional[EmbeddingQuantOptions] = None
    quantize_linear: Optional[DynamicQuantLinearOptions] = None


def _get_pt2e_quantization_params(args) -> Optional[PT2EQuantOptions]:
    if args.pt2e_quantize is None:
        return None
    if args.quantization_mode:
        raise ValueError("Cannot specify both --quantization_mode and --pt2e_quantize")

    quantization_options = args.pt2e_quantize.split(",")
    quantization_options = [option.strip() for option in quantization_options]
    # This can really be improved significantly.
    # Hopefully we dont release this in its current form.
    # Just using this for quick experiments.
    quant_options = None
    if "embedding" in quantization_options:
        quant_options = quant_options or PT2EQuantOptions()
        quant_options.quantize_embedding = EmbeddingQuantOptions()
    if (
        "xnnpack_dynamic" in quantization_options
        and "xnnpack_dynamic_qc4" in quantization_options
    ):
        raise RuntimeError(
            "For dynamic linear quantization via xnnpack quantizer you can chose only qc8 or qc4 option, not both."
        )
    if (
        "xnnpack_dynamic" in quantization_options
        or "xnnpack_dynamic_qc4" in quantization_options
    ):
        quant_options = quant_options or PT2EQuantOptions()
        quant_options.quantize_linear = DynamicQuantLinearOptions()
        if "xnnpack_dynamic_qc4" in quantization_options:
            quant_options.quantize_linear.is_qc4 = True

    return quant_options


# TODO: move args is used only get so_file. Refactor this
def get_pt2e_quantizers(
    quant_params: Optional[PT2EQuantOptions], args
) -> List[Quantizer]:
    """
    Get a list of quantizers from quantization params
    Args:
        args: quant params
    Returns:
        A list of quantizers to pass into LlamaBuilder.
    """

    def check_embedding_byte_registered():
        try:
            _ = torch.ops.quantized_decomposed.embedding_byte.out
        except AttributeError:
            if args.so_library:
                print(f"Loading library {args.so_library}")
                torch.ops.load_library(args.so_library)
            else:
                raise RuntimeError(
                    "Need to specify shared library path to register quantized ops (and their out variants) into EXIR.\n"
                    "Follow the following steps to build the needed lib via cmake.\n"
                    'Use `python -c "import torch as _; print(_.__path__)"` to find where torch package is installed.\n'
                    "Set that as TORCH_PACKAGE_DIR.\n"
                    "Then from root executorch dir do the following:\n"
                    "rm -rf cmake-out && mkdir cmake-out && (cd cmake-out && cmake -DBUCK2=<path-to-buck2> -DCMAKE_PREFIX_PATH=$TORCH_PACKAGE_DIR -DREGISTER_QUANTIZED_OPS=ON ..) && cmake --build . -j16\n"
                    'To find the location of the lib: find cmake-out -name "libquantized_ops_aot_lib*"\n'
                    "Then specify the said library via -s <path to libquantized_ops_aot_lib.so\n"
                )

    quantizers = []
    if quant_params is not None and quant_params.quantize_embedding is not None:
        logging.info("Apply PT2E embedding quantization.")
        check_embedding_byte_registered()
        quantizers.append(EmbeddingQuantizer())
    if quant_params is not None and quant_params.quantize_linear is not None:
        logging.info("Apply PT2E dynamic linear quantization.")
        dynamic_quantizer = XNNPACKQuantizer()
        assert quant_params.quantize_linear is not None
        if not quant_params.quantize_linear.is_per_channel:
            raise ValueError(
                "At the moment only per channel weight quantization is supported."
            )
        if quant_params.quantize_linear.is_qc4:
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True, weight_qmin=-8, weight_qmax=7
            )
        else:
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
        dynamic_quantizer.set_global(operator_config_dynamic)
        quantizers.append(dynamic_quantizer)
    return quantizers


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


def quantize(
    model: torch.nn.Module,
    qmode: str,
    activation_dtype: Optional[DType],
    checkpoint_path: Optional[Path] = None,
    # following arguments only available when setting int4 quantization.
    group_size: int = 128,
    # following arguments only used for GPTQ
    calibration_tasks: Optional[list] = None,
    calibration_limit: int = 5,
    calibration_seq_length: int = 100,
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

    if checkpoint_path is None:
        checkpoint_path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")

    if calibration_tasks is None:
        calibration_tasks = ["wikitext"]

    if qmode == "int8":
        # Add quantization mode options here: group size, bit width, etc.
        return WeightOnlyInt8QuantHandler(model).quantized_model()
    elif qmode == "8da4w":
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        model = Int8DynActInt4WeightQuantizer(precision=torch_dtype).quantize(model)
        print("quantized model:", model)
        return model
    elif qmode == "8da4w-gptq":
        from torchao.quantization.quant_api import Int8DynActInt4WeightGPTQQuantizer

        if tokenizer_path is None:
            tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
        )
        gptq_quantizer = Int8DynActInt4WeightGPTQQuantizer(
            tokenizer,
            blocksize,
            percdamp,
            group_size,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs,
        )
        model = gptq_quantizer.quantize(model)
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
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=[],
        help="Tasks for GPTQ calibration",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=5,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=2048,
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
        "-G", "--group_size", default=None, help="group_size for weight quantization"
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


def canonical_path(path: str, *, dir: bool = False) -> str:

    print(f"creating canonical path for {path}")
    if not path.startswith("par:"):
        return path

    if not IS_FBCODE:
        print("not FBCODE")
        return path[4:]
    else:
        return_val = pkg_resources.resource_filename(pkg_name, path[4:])
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
    checkpoint_path = canonical_path(args.checkpoint)
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
        transforms.append(
            partial(
                quantize,
                qmode=args.quantization_mode,
                activation_dtype=dtype_override,
                checkpoint_path=(
                    Path(path) if (path := args.checkpoint) is not None else None
                ),
                tokenizer_path=(
                    Path(path) if (path := args.tokenizer_path) is not None else None
                ),
                group_size=args.group_size,
                calibration_tasks=args.calibration_tasks,
                calibration_limit=args.calibration_limit,
                calibration_seq_length=args.calibration_seq_length,
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

    return (
        load_llama_model(
            checkpoint=checkpoint_path,
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

    builder_exported_to_edge = _prepare_for_llama_export(
        modelname, args
    ).export_to_edge(quantizers)

    # to_backend
    partitioner = None
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioner = XnnpackDynamicallyQuantizedPartitioner()
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        # Following changes due to.
        # 1. We need dynamically quantized partitioner for both pt2e_quantize options
        #    as well as "qmode 8da4w" which is also dynamic quantizes linear layers.
        # 2. XNNPACK partitioner seems to result in seg fault for non dqlinear ops.
        partitioner = XnnpackDynamicallyQuantizedPartitioner()
        # partitioner = XnnpackPartitioner()
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        assert (
            args.dtype_override == "fp32" or args.dtype_override is None
        ), "Vulkan backend does not support non fp32 dtypes at the moment"
        assert (
            args.quantization_mode is None
        ), "Vulkan backend does not support quantization at the moment"

        partitioner = VulkanPartitioner()
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
        partitioner = MPSPartitioner(compile_specs)
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
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`
        partitioner = CoreMLPartitioner(
            skip_ops_for_coreml_delegation=None, compile_specs=compile_specs
        )
        modelname = f"coreml_{modelname}"

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        builder = builder_exported_to_edge.to_backend(partitioner).to_executorch()

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord(
                etrecord_path="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        builder = builder_exported_to_edge.to_backend(partitioner).to_executorch()

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
