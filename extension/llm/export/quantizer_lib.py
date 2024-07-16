# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is for PT2E quantization.

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


def get_pt2e_quantization_params(
    pt2e_quantize: Optional[str] = None,
    quantization_mode: Optional[str] = None,
) -> Optional[PT2EQuantOptions]:
    if pt2e_quantize is None:
        return None
    if quantization_mode:
        raise ValueError("Cannot specify both quantization_mode and pt2e_quantize")

    quantization_options = pt2e_quantize.split(",")
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


def get_pt2e_quantizers(
    quant_params: Optional[PT2EQuantOptions],
    so_library: Optional[str] = None,
) -> List[Quantizer]:
    """
    Get a list of quantizers from quantization params
    Args:
        quant_params: PT2E quantization options.
    Returns:
        A list of quantizers to pass into LlamaBuilder.
    """

    def check_embedding_byte_registered():
        try:
            _ = torch.ops.quantized_decomposed.embedding_byte.out
        except AttributeError:
            if so_library:
                print(f"Loading library {so_library}")
                torch.ops.load_library(so_library)
            else:
                raise RuntimeError(
                    "Need to specify shared library path to register quantized ops (and their out variants) into EXIR.\n"
                    "Follow the following steps to build the needed lib via cmake.\n"
                    'Use `python -c "import torch as _; print(_.__path__)"` to find where torch package is installed.\n'
                    "Set that as TORCH_PACKAGE_DIR.\n"
                    "Then from root executorch dir do the following:\n"
                    "rm -rf cmake-out && mkdir cmake-out && (cd cmake-out && cmake -DBUCK2=<path-to-buck2> -DCMAKE_PREFIX_PATH=$TORCH_PACKAGE_DIR -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON ..) && cmake --build . -j16\n"
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


def get_qnn_quantizer(
    pt2e_quantize: str,
    quantization_mode: Optional[str] = None,
):
    try:
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.quantizer.quantizer`
        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_16a4w_qnn_ptq_config,
            get_default_16bit_qnn_ptq_config,
            QnnQuantizer,
            QuantDtype,
        )
        from torch.ao.quantization.observer import MinMaxObserver

    except ImportError:
        raise ImportError(
            "Please install the Qualcomm backend follwing https://pytorch.org/executorch/main/build-run-qualcomm.html"
        )

    def annotate_matmul_16a8w(gm: torch.fx.GraphModule) -> None:
        """
        This function is specific for matmul op 16a8w.
        """
        from executorch.backends.qualcomm.quantizer.quantizer import (
            get_16a8w_qnn_ptq_config,
            get_default_8bit_qnn_ptq_config,
            QuantizationConfig,
        )
        from executorch.backends.qualcomm.quantizer.utils import QUANT_ANNOTATION_KEY
        from torch.ao.quantization.quantizer import (
            QuantizationAnnotation,
            SharedQuantizationSpec,
        )
        from torch.fx import Node

        def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
            input_qspec_map = {}
            input_act = node.args[0]
            input_spec = quantization_config.input_activation
            input_qspec_map[input_act] = input_spec

            input_act1 = node.args[1]
            input_spec1 = quantization_config.weight
            input_qspec_map[input_act1] = input_spec1

            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )
        def annotate_index_put(node: Node, quantization_config: QuantizationConfig) -> None:
            input = node.args[0]
            value = node.args[2]

            input_qspec_map = {}
            input_qspec_map[input] = quantization_config.input_activation
            input_qspec_map[value] = SharedQuantizationSpec((input, node))

            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=SharedQuantizationSpec((input, node)),
                _annotated=True,
            )

        def annotate_single_in_single_out(
            node: Node, quantization_config: QuantizationConfig
        ) -> None:

            input_qspec_map = {}
            input_act = node.args[0]
            input_qspec_map[input_act] = quantization_config.input_activation

            node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=quantization_config.output_activation,
                _annotated=True,
            )

        def annotate_matmul_input1(node: Node):
            quantization_config_8a8w = get_default_8bit_qnn_ptq_config(act_symmetric=True)
            while isinstance(node, Node) and node.op == "call_function":
                if node.target == torch.ops.aten.index_put_.default:
                    annotate_index_put(node, quantization_config_8a8w)
                    break
                annotate_single_in_single_out(node, quantization_config_8a8w)
                node = node.args[0]

        quantization_config_16a8w = get_16a8w_qnn_ptq_config()

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
                annotate_matmul(node, quantization_config_16a8w)
                annotate_matmul_input1(node.args[1])
    
    backend, quant_config = pt2e_quantize.split("_")
    assert (
        backend == "qnn"
    ), f"The quantization config is for backend {backend} instead of qnn."
    qnn_quantizer = QnnQuantizer()
    qnn_quantizer.set_per_channel_conv_quant(enable=True)
    qnn_quantizer.set_per_channel_linear_quant(enable=True)
    # more custom quantization are supported including 16a4w etc. default to 8bit quantized

    if quant_config == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
        custom_annotations = ()
    elif quant_config == "16a16w":
        quant_dtype = QuantDtype.use_16a16w
        qnn_quantizer.add_16bit_quant_ops(qnn_quantizer.SUPPORTED_OPS)
        qnn_quantizer.set_bit16_op_quant_config(
            get_default_16bit_qnn_ptq_config(act_observer=MinMaxObserver)
        )
        custom_annotations = ()
    elif quant_config == "16a4w":
        quant_dtype = QuantDtype.use_16a4w
        qnn_quantizer.add_16bit_quant_ops(qnn_quantizer.SUPPORTED_OPS)
        qnn_quantizer.set_bit16_op_quant_config(
            get_16a4w_qnn_ptq_config(act_observer=MinMaxObserver)
        )
        qnn_quantizer.set_per_channel_weight_dtype(weight_dtype_for_16bit_act="int4")
        custom_annotations = (annotate_matmul_16a8w, )
    else:
        raise AssertionError(
            f"No support for quant type {quant_config}. Support 8a8w, 16a16w and 16a4w."
        )

    assert (
        quantization_mode is None
    ), "Currently qnn backend only supports QnnQuantizer via pt2e flow"
    qnn_quantizer.add_custom_quant_annotations(custom_annotations)
    return qnn_quantizer, quant_dtype


def get_coreml_quantizer(pt2e_quantize: str):
    try:
        from coremltools.optimize.torch.quantization.quantization_config import (
            LinearQuantizerConfig,
            QuantizationScheme,
        )

        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.apple.coreml.quantizer`.
        from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
    except ImportError:
        raise ImportError(
            "Please install the CoreML backend follwing https://pytorch.org/executorch/main/build-run-coreml.html"
        )

    if pt2e_quantize == "coreml_8a_c8w":
        config = LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": QuantizationScheme.affine,
                    "activation_dtype": torch.quint8,
                    "weight_dtype": torch.qint8,
                    "weight_per_channel": True,
                }
            }
        )
        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `apple`.
        quantizer = CoreMLQuantizer(config)

    elif pt2e_quantize in ("coreml_c4w", "coreml_8a_c4w"):
        raise NotImplementedError("4-bit Core ML quantizer is still under development")

    elif pt2e_quantize == "coreml_baseline_8a_c8w":
        config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=False
        )
        quantizer = XNNPACKQuantizer().set_global(config)

    elif pt2e_quantize == "coreml_baseline_8a_c4w":
        config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=False, weight_qmin=-8, weight_qmax=7
        )
        quantizer = XNNPACKQuantizer().set_global(config)

    else:
        raise ValueError(f"Unsupported Core ML quantizer specification {pt2e_quantize}")

    return quantizer
