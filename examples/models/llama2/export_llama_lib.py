# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import argparse
import logging
import shlex
from dataclasses import dataclass

from functools import partial
from pathlib import Path
from typing import List, Optional

import pkg_resources
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    XnnpackPartitioner,
)
from executorch.util.activation_memory_profiler import generate_memory_trace
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.embedding_quantizer import EmbeddingQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from .builder import DType, load_llama_model, WeightType

from .quantize import (
    EmbeddingOnlyInt8QuantHandler,
    Int8DynActInt4WeightQuantHandler,
    WeightOnlyInt8QuantHandler,
)

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


def quantize(
    model: torch.nn.Module,
    qmode: str,
    activation_dtype: Optional[DType],
) -> torch.nn.Module:
    """
    Quantizes a model by converting all weights to int8.
    Args:
        model: A model to quantize.
        qmode: quantization mode, e.g. int8, int4
    Returns:
        A quantized model.
    """
    if activation_dtype is not None:
        torch_dtype = activation_dtype.to_torch_dtype()
    else:
        torch_dtype = torch.float16

    if qmode == "int8":
        # Add quantization mode options here: group size, bit width, etc.
        return WeightOnlyInt8QuantHandler(model).quantized_model()
    elif qmode == "int4":
        model_int4 = Int8DynActInt4WeightQuantHandler(
            model, activation_precision=torch_dtype
        ).quantized_model()
        print("quantized model:", model_int4)
        return model_int4
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
    parser.add_argument(
        "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
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
        help="Use PT2E quantization. Comma separated options. e.g. xnnpack_dynamic (for per channel 8 bit weight), xnnpack_dynamic_qc4 (for per channel 4 bit weight), embedding.",
    )
    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=str,
        default=None,
        choices=["int8", "int4"],
        help="type of quantization",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=f"{ckpt_dir}/params/demo_rand_params.pth",
        help="checkpoint path",
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
    parser.add_argument("-G", "--groupsize", default=None, help="specify the groupsize")

    parser.add_argument(
        "-d",
        "--dtype-override",
        default=None,
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp16, fp32",
    )
    parser.add_argument("-2", "--fairseq2", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-X", "--xnnpack", action="store_true")

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


def _export_llama(modelname, args) -> str:  # noqa: C901
    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint)
    params_path = canonical_path(args.params)
    output_dir_path = canonical_path(args.output_dir, dir=True)
    modelname = "llama2"
    weight_type = WeightType.FAIRSEQ2 if args.fairseq2 else WeightType.LLAMA

    # dtype override
    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
    else:
        dtype_override = DType["fp16"] if args.quantization_mode == "int4" else None

    # source transforms
    transforms = []
    if args.quantized_ckpt or args.quantization_mode:
        modelname = f"{modelname}_q"
        transforms.append(
            partial(
                quantize, qmode=args.quantization_mode, activation_dtype=dtype_override
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

    # export_to_edge
    pt2e_quant_params = _get_pt2e_quantization_params(args)
    quantizers = get_pt2e_quantizers(pt2e_quant_params, args)

    # to_backend
    partitioners = {}
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioners[
            XnnpackDynamicallyQuantizedPartitioner.__name__
        ] = XnnpackDynamicallyQuantizedPartitioner()
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        partitioners[XnnpackPartitioner.__name__] = XnnpackPartitioner()
        modelname = f"xnnpack_{modelname}"

    # TODO: remove this after xnnpack delegation is ready
    if args.quantization_mode == "int4":
        raise Exception(
            "some quantized ops should be lowered to xnnpack, but xnnpack delegate is not ready yet"
        )

    builder = (
        load_llama_model(
            checkpoint=checkpoint_path,
            params_path=params_path,
            use_kv_cache=args.use_kv_cache,
            use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
            weight_type=weight_type,
            verbose=args.verbose,
        )
        .set_output_dir(output_dir_path)
        .set_metadata(args.metadata)
        .source_transform(transforms)
        .to_dtype(dtype_override)
        .export_to_edge(quantizers)
        .to_backend(partitioners)
        .to_executorch()
    )

    if args.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_fp16"

    if args.use_kv_cache:
        modelname = f"{modelname}_kv"

    builder.save_to_pte(modelname)
    output_file = f"{builder.output_dir}/{modelname}.pte"

    return output_file
