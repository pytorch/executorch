# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List, Optional, Tuple

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer import Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    quantize_module,
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program

from torchsr import transforms


def get_dataset(
    root_dir: str,
    calinum=100,
    transform_compose: Optional[transforms.Compose] = None,
) -> Tuple:
    """
    Generate test data from B100 dataset for quantization model

    :param root_dir: Dir of dataset. The real dataset should be in root_dir/SRBenchmarks/benchmark/
    :param dataset_name: data_set name
    :param testnum: Number of test data. Default 500
    :param transform_compose: Transforms to be applied to data.
        Default:
        transform_compose = transforms.Compose(
            [transforms.ToTensor()] # Convert Pillows Image to tensor
        )
    :type root_dir: str
    :type calinum: int
    :type testnum: int
    :type transform_compose: transforms.Compose | None
    :return: (example_input, cali_data, test_data)
    """

    class SrResize:
        def __init__(self, expected_size: List[List[int]]):
            self.expected_size = expected_size

        def __call__(self, x):
            return (
                x[0].resize(self.expected_size[0]),
                x[1].resize(self.expected_size[1]),
            )

    class SrUnsqueeze:
        def __call__(self, x):
            return (
                x[0].unsqueeze(0),
                x[1].unsqueeze(0),
            )

    if not transform_compose:
        transform_compose = transforms.Compose(
            [
                SrResize([[448, 448], [224, 224]]),
                transforms.ToTensor(),  # Convert Pillows Image to tensor
                SrUnsqueeze(),
            ]
        )
    from torchsr.datasets import B100

    dataset = B100(root=root_dir, transform=transform_compose, scale=2)
    example_data = [(dataset[i][1],) for i in range(min(calinum, len(dataset)))]
    return example_data


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
        help=("path to the validation folder of B100"),
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
        default="./edsr",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "edsr"
    instance = EdsrModel()
    model = EdsrModel().get_eager_model().eval()
    assert args.calibration_number
    if args.dataset:
        inputs = get_dataset(
            root_dir=f"{args.dataset}",
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
