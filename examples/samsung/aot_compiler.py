# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    to_edge_transform_and_lower_to_enn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SUPPORT_MODEL_NAMES = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--chipset",
        required=True,
        help="Samsung chipset, i.e. E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model name. Valid ones: {SUPPORT_MODEL_NAMES}",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")

    args = parser.parse_args()

    if args.model_name not in SUPPORT_MODEL_NAMES and args.quantize:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not quantizable right now, "
            "please contact executorch team if you want to learn why or how to support "
            "quantization for the requested model"
            f"Available models are {SUPPORT_MODEL_NAMES}."
        )

    model, example_inputs, dynamic_shapes, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )
    assert (
        dynamic_shapes is None
    ), "enn backend doesn't support dynamic shapes currently."

    model = model.eval()

    if args.quantize:
        raise NotImplementedError("Quantizer is under developing...")
    else:
        # TODO(anyone) Remove the judgement after quantizer work fine or judge it in other ways
        # raise AssertionError("Only support s8/fp16/s16")
        pass

    # logging.info(f"Exported graph:\n{edge.exported_program().graph}")
    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]
    edge = to_edge_transform_and_lower_to_enn(
        model, example_inputs, compile_specs=compile_specs
    )

    # Save
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    quant_tag = "q8" if args.quantize else "fp32"
    model_name = f"{args.model_name}_exynos_{quant_tag}"
    save_pte_program(exec_prog, model_name, args.output_dir)
