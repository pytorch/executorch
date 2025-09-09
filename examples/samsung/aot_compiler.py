# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import logging
import os

import torch

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

SUPPORT_MODEL_NAMES = [
    "mv2",
    "ic3",
    "ic4",
    "resnet18",
    "resnet50",
    "mv3",
    "edsr",
    "dl3",
    "vit",
]


def save_tensors(tensors, prefix, artifact_dir):
    if isinstance(tensors, tuple):
        for index, output in enumerate(tensors):
            save_path = prefix + "_" + str(index) + ".bin"
            output.detach().numpy().tofile(os.path.join(artifact_dir, save_path))
    elif isinstance(tensors, torch.Tensor):
        tensors.detach().numpy().tofile(os.path.join(artifact_dir, prefix + ".bin"))
    elif isinstance(tensors, collections.OrderedDict):
        for index, output in enumerate(tensors.values()):
            save_path = prefix + "_" + str(index) + ".bin"
            output.detach().numpy().tofile(os.path.join(artifact_dir, save_path))
    else:
        logging.warning("Unsupported type (", type(tensors), ") skip saving tensor. ")


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

    if args.model_name not in SUPPORT_MODEL_NAMES:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not support yet. "
            "In the near future, more example models will be supported. Currently, "
            f"Available models are {SUPPORT_MODEL_NAMES}."
        )

    model, example_inputs, dynamic_shapes, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )
    assert (
        dynamic_shapes is None
    ), "enn backend doesn't support dynamic shapes currently."

    model = model.eval()
    outputs = model(*example_inputs)

    print("start start ...")

    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]
    edge = to_edge_transform_and_lower_to_enn(
        model, example_inputs, compile_specs=compile_specs
    )

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    model_name = f"{args.model_name}_exynos_fp32"
    save_pte_program(exec_prog, model_name, args.output_dir)

    save_tensors(example_inputs, f"{args.model_name}_input", args.output_dir)
    save_tensors(outputs, f"{args.model_name}_output", args.output_dir)
