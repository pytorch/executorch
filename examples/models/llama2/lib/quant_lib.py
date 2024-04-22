# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is for PT2E quantization. For source-transformation quantize, please modify source_transformation/quantize.py

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.embedding_quantizer import EmbeddingQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

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
                    "rm -rf cmake-out && mkdir cmake-out && (cd cmake-out && cmake -DBUCK2=<path-to-buck2> -DCMAKE_PREFIX_PATH=$TORCH_PACKAGE_DIR -DEXECUTORCH_BUILD_QUANTIZED_OPS_AOT=ON ..) && cmake --build . -j16\n"
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


def get_qnn_quantizer(args):
    try:
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.quantizer.quantizer`
        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_16a4w_qnn_ptq_config,
            get_default_16bit_qnn_ptq_config,
            QnnQuantizer,
            QuantDtype,
        )

    except ImportError:
        raise ImportError(
            "Please install the Qualcomm backend follwing https://pytorch.org/executorch/main/build-run-qualcomm.html"
        )

    backend, quant_config = args.pt2e_quantize.split("_")
    assert (
        backend == "qnn"
    ), f"The quantization config is for backend {backend} instead of qnn."
    qnn_quantizer = QnnQuantizer()
    # more custom quantization are supported including 16a4w etc. default to 8bit quantized
    custom_annotations = ()
    if quant_config == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
        pass
    elif quant_config == "16a16w":
        quant_dtype = QuantDtype.use_16a16w
        qnn_quantizer.add_16bit_quant_ops(qnn_quantizer.SUPPORTED_OPS)
        qnn_quantizer.set_bit16_op_quant_config(get_default_16bit_qnn_ptq_config())
    elif quant_config == "16a4w":
        quant_dtype = QuantDtype.use_16a4w
        qnn_quantizer.add_16bit_quant_ops(qnn_quantizer.SUPPORTED_OPS)
        qnn_quantizer.set_bit16_op_quant_config(get_16a4w_qnn_ptq_config())
        qnn_quantizer.set_per_channel_weight_dtype(weight_dtype_for_16bit_act="int4")
    else:
        raise AssertionError(
            f"No support for quant type {quant_config}. Support 8a8w, 16a16w and 16a4w."
        )

    assert (
        args.quantization_mode is None
    ), "Currently qnn backend only supports QnnQuantizer via pt2e flow"
    qnn_quantizer.add_custom_quant_annotations(custom_annotations)
    return qnn_quantizer, quant_dtype
