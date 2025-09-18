# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Optional

import torch
import torchvision.transforms.v2 as vision_transform_v2

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer import Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    quantize_module,
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet50Model
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from torchvision.datasets import VOCSegmentation


def get_dataset(
    data_dir: str,
    calinum=100,
    input_transform_compose: Optional[vision_transform_v2.Compose] = None,
    target_transform_compose: Optional[vision_transform_v2.Compose] = None,
):
    if not input_transform_compose:
        input_transform_compose = vision_transform_v2.Compose(
            [
                vision_transform_v2.Resize([224, 224]),
                vision_transform_v2.ToImage(),
                vision_transform_v2.ToDtype(torch.float32, scale=True),
                vision_transform_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                vision_transform_v2.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
            ]
        )
    if not target_transform_compose:
        target_transform_compose = vision_transform_v2.Compose(
            [
                vision_transform_v2.Resize([224, 224]),
                vision_transform_v2.ToImage(),
                vision_transform_v2.ToDtype(torch.long, scale=False),
                vision_transform_v2.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
            ]
        )
    voc_dataset = VOCSegmentation(
        data_dir,
        "2012",
        "val",
        transform=input_transform_compose,
        target_transform=target_transform_compose,
    )
    example_input = [
        (voc_dataset[i][0],) for i in range(min(calinum, len(voc_dataset)))
    ]
    return example_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--chipset",
        default="E9955",
        help="Samsung chipset, i.e. E9945, E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help=("path to the validation folder of VOC dataset. "),
        type=str,
    )

    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        help=("Quantizaiton precision. If not set, the model will not be quantized."),
        choices=[None, "A8W8"],
        type=str,
    )

    parser.add_argument(
        "-cn",
        "--calibration_number",
        default=100,
        help=(
            "Assign the number of data you want "
            "to use for calibrating the quant params."
        ),
        type=int,
    )

    parser.add_argument(
        "--dump",
        default=False,
        const=True,
        nargs="?",
        help=("Whether to dump all outputs. If not set, we only dump pte."),
        type=bool,
    )

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. ",
        default="./deeplab_v3",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "deeplab_v3"
    instance = DeepLabV3ResNet50Model()
    model = DeepLabV3ResNet50Model().get_eager_model().eval()
    assert args.calibration_number
    if args.dataset:
        inputs = get_dataset(
            data_dir=f"{args.dataset}",
            calinum=args.calibration_number,
        )
    else:
        inputs = [instance.get_example_inputs() for _ in range(args.calibration_number)]

    test_in = inputs[0]
    float_out = model(*test_in)

    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]

    if args.precision:
        model = quantize_module(
            model, inputs[0], inputs, getattr(Precision, args.precision)
        )
        quant_out = model(*test_in)

    edge_prog = to_edge_transform_and_lower_to_enn(
        model, inputs[0], compile_specs=compile_specs
    )

    edge = edge_prog.to_backend(EnnPartitioner(compile_specs))
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    save_pte_program(exec_prog, pte_filename, os.path.join(f"{args.artifact}"))

    if args.dump:
        save_tensors(test_in, "float_in", args.artifact)
        save_tensors(float_out, "float_out", args.artifact)
        if args.precision:
            save_tensors(quant_out, "quant_out", args.artifact)
