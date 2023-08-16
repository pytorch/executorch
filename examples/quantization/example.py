# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy

import torch
import torch._export as export
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

# TODO: maybe move this to examples/export/utils.py?
# from ..export.export_example import export_to_ff

from ..models import MODEL_NAME_TO_MODEL


def quantize(model_name, model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    m = model.eval()
    m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
    print("original model:", m)
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(m, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    print("quantized model:", m)
    # make sure we can export to flat buffer
    # Note: this is not working yet due to missing out variant ops for quantize_per_tensor/dequantize_per_tensor ops
    # aten = export_to_ff(model_name, m, copy.deepcopy(example_inputs))


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
    print("m:", m)
    print("m_fx:", m_fx)
    print("prepare sqnr:", compute_sqnr(after_prepare_result, after_prepare_result_fx))
    assert compute_sqnr(after_prepare_result, after_prepare_result_fx) > 100
    print("quant diff max:", torch.max(after_quant_result - after_quant_result_fx))
    assert torch.max(after_quant_result - after_quant_result_fx) < 1e-1
    print("quant sqnr:", compute_sqnr(after_quant_result, after_quant_result_fx))
    assert compute_sqnr(after_quant_result, after_quant_result_fx) > 30


if __name__ == "__main__":
    # Note: for mv3, the mul op is not supported in XNNPACKQuantizer, that could be supported soon
    QUANT_MODEL_NAME_TO_MODEL = {
        name: MODEL_NAME_TO_MODEL[name] for name in ["linear", "add", "add_mul", "mv2"]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(QUANT_MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-ve",
        "--verify",
        action="store_true",
        required=False,
        default=False,
        help="flag for verifying XNNPACKQuantizer against fx graph mode quantization",
    )

    args = parser.parse_args()

    if not args.verify and args.model_name not in QUANT_MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not quantizable right now, "
            "please contact executorch team if you want to learn why or how to support "
            "quantization for the requested model"
            f"Available models are {list(QUANT_MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = MODEL_NAME_TO_MODEL[args.model_name]()

    if args.verify:
        verify_xnnpack_quantizer_matching_fx_quant_model(
            args.model_name, model, example_inputs
        )

    quantize(args.model_name, model, example_inputs)
    print("finished")
