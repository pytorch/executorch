# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quantization Related Logic used in LLama_lib's

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from sentencepiece import SentencePieceProcessor

from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.embedding_quantizer import EmbeddingQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from .builder import DType
from .quantize import WeightOnlyInt8QuantHandler


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


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
                    "rm -rf cmake-out && mkdir cmake-out && (cd cmake-out && cmake -DBUCK2=<path-to-buck2> -DCMAKE_PREFIX_PATH=$TORCH_PACKAGE_DIR -DEXECUTORCH_BUILD_QUANTIZED=ON ..) && cmake --build . -j16\n"
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
