# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import json
import numpy as np
import argparse

import torch
from executorch.backends.mediatek import Precision
from executorch.examples.models.wav2letter import Wav2LetterModel
from aot_utils.oss_utils.utils import (
    build_executorch_binary,
    make_output_dir,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./wav2letter",
        default="./wav2letter",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "wav2letter_mtk"
    model = Wav2LetterModel()
    instance = model.get_eager_model()
    inputs = model.get_example_inputs()

    build_executorch_binary(
        instance.eval(),
        inputs,
        f"{args.artifact}/{pte_filename}",
        [inputs],
        quant_dtype=Precision.A8W8,
        skip_op_name = {
            "aten_convolution_default",
            "aten_convolution_default_1",
            "aten_convolution_default_9",
            "aten__log_softmax_default",
        },
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list.txt"
    with open(input_list_file, "w") as f:
        f.write("input_0_0.bin")
        f.flush()
    file_name = f"{args.artifact}/input_0_0.bin"
    inputs[0].detach().numpy().tofile(file_name)
    file_name = f"{args.artifact}/golden_0_0.bin"
    golden = instance(inputs[0])
    golden.detach().numpy().tofile(file_name)

