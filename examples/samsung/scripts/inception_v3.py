# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer import Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    quantize_module,
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.models.inception_v3 import InceptionV3Model
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program


def get_dataset(dataset_path, data_size):
    from torchvision import datasets, transforms

    image_shape = (256, 256)
    crop_size = 224
    shuffle = True

    def get_data_loader():
        preprocess = transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        imagenet_data = datasets.ImageFolder(dataset_path, transform=preprocess)
        return torch.utils.data.DataLoader(
            imagenet_data,
            shuffle=shuffle,
        )

    # prepare input data
    inputs, targets, input_list = [], [], ""
    data_loader = get_data_loader()
    for index, data in enumerate(data_loader):
        if index >= data_size:
            break
        feature, target = data
        inputs.append((feature,))
        targets.append(target)
        input_list += f"input_{index}_0.bin\n"

    return inputs, targets, input_list


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
        help=(
            "path to the validation folder of ImageNet dataset. "
            "e.g. --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)"
        ),
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
        default="./inception_v3",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "inception_v3"
    instance = InceptionV3Model()
    model = InceptionV3Model().get_eager_model().eval()
    assert args.calibration_number
    if args.dataset:
        inputs, targets, input_list = get_dataset(
            dataset_path=f"{args.dataset}",
            data_size=args.calibration_number,
        )
    else:
        inputs = [instance.get_example_inputs() for _ in range(args.calibration_number)]
        target = None
        input_list = None

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
