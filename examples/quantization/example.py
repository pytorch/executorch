# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import logging
import time

import torch
import torch._export as export
from executorch.exir import EdgeCompileConfig
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (  # @manual
    default_per_channel_symmetric_qnnpack_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import get_executorch_backend_config
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from ..export.utils import export_to_edge, save_pte_program
from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from ..recipes.xnnpack_optimization import MODEL_NAME_TO_OPTIONS

from .utils import quantize


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def verify_xnnpack_quantizer_matching_fx_quant_model(model_name, model, example_inputs):
    """This is a verification against fx graph mode quantization flow as a sanity check"""

    model.eval()
    m_copy = copy.deepcopy(model)
    m = model

    # 1. pytorch 2.0 export quantization flow (recommended/default flow)
    m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
    quantizer = XNNPACKQuantizer()
    quantization_config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(quantization_config)
    m = prepare_pt2e(m, quantizer)
    # calibration
    after_prepare_result = m(*example_inputs)
    m = convert_pt2e(m)
    after_quant_result = m(*example_inputs)

    # 2. the previous fx graph mode quantization reference flow
    qconfig = default_per_channel_symmetric_qnnpack_qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    backend_config = get_executorch_backend_config()
    m_fx = prepare_fx(
        m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
    )
    after_prepare_result_fx = m_fx(*example_inputs)
    m_fx = _convert_to_reference_decomposed_fx(m_fx, backend_config=backend_config)
    after_quant_result_fx = m_fx(*example_inputs)

    # 3. compare results
    # NB: this check is more useful for QAT since for PTQ we are only inserting observers that does not change the
    # output of a model, so it's just testing the numerical difference for different captures in PTQ
    # for QAT it is also testing whether the fake quant placement match or not
    # not exactly the same due to capture changing numerics, but still really close
    logging.info(f"m: {m}")
    logging.info(f"m_fx: {m_fx}")
    logging.info(
        f"prepare sqnr: {compute_sqnr(after_prepare_result, after_prepare_result_fx)}"
    )
    assert compute_sqnr(after_prepare_result, after_prepare_result_fx) > 100
    logging.info(
        f"quant diff max: {torch.max(after_quant_result - after_quant_result_fx)}"
    )
    assert torch.max(after_quant_result - after_quant_result_fx) < 1e-1
    logging.info(
        f"quant sqnr: {compute_sqnr(after_quant_result, after_quant_result_fx)}"
    )
    assert compute_sqnr(after_quant_result, after_quant_result_fx) > 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_OPTIONS.keys())}",
    )
    parser.add_argument(
        "-ve",
        "--verify",
        action="store_true",
        required=False,
        default=False,
        help="flag for verifying XNNPACKQuantizer against fx graph mode quantization",
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        help="shared library for quantized operators",
    )

    args = parser.parse_args()
    # See if we have quantized op out variants registered
    has_out_ops = True
    try:
        op = torch.ops.quantized_decomposed.add.out
    except AttributeError:
        logging.info("No registered quantized ops")
        has_out_ops = False
    if not has_out_ops:
        if args.so_library:
            torch.ops.load_library(args.so_library)
        else:
            raise RuntimeError(
                "Need to specify shared library path to register quantized ops (and their out variants) into"
                "EXIR. The required shared library is defined as `quantized_ops_aot_lib` in "
                "kernels/quantized/CMakeLists.txt if you are using CMake build, or `aot_lib` in "
                "kernels/quantized/targets.bzl for buck2. One example path would be cmake-out/kernels/quantized/"
                "libquantized_ops_aot_lib.[so|dylib]."
            )
    if not args.verify and args.model_name not in MODEL_NAME_TO_OPTIONS:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not quantizable right now, "
            "please contact executorch team if you want to learn why or how to support "
            "quantization for the requested model"
            f"Available models are {list(MODEL_NAME_TO_OPTIONS.keys())}."
        )

    start = time.perf_counter()
    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )
    end = time.perf_counter()
    # logging.info(f"Model init time: {end - start}s")
    if args.verify:
        start = time.perf_counter()
        verify_xnnpack_quantizer_matching_fx_quant_model(
            args.model_name, model, example_inputs
        )
        end = time.perf_counter()
        # logging.info(f"Verify time: {end - start}s")

    model = model.eval()
    # pre-autograd export. eventually this will become torch.export
    model = export.capture_pre_autograd_graph(model, example_inputs)
    start = time.perf_counter()
    quantized_model = quantize(model, example_inputs)
    end = time.perf_counter()
    # logging.info(f"Quantize time: {end - start}s")

    # TODO[T163161310]: takes a long time to export to exec prog and save inception_v4 quantized model
    if args.model_name != "ic4":

        start = time.perf_counter()
        edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
        edge_m = export_to_edge(
            quantized_model, example_inputs, edge_compile_config=edge_compile_config
        )
        end = time.perf_counter()

        start = time.perf_counter()
        prog = edge_m.to_executorch(None)
        save_pte_program(prog.buffer, f"{args.model_name}_quantized")
        end = time.perf_counter()
    logging.info("finished")
