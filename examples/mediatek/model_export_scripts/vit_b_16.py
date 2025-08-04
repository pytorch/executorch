# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse

import torch
from aot_utils.oss_utils.utils import build_executorch_binary
from executorch.backends.mediatek import Precision
from executorch.examples.models.torchvision_vit import TorchVisionViTModel


class NhwcWrappedModel(torch.nn.Module):
    def __init__(self):
        super(NhwcWrappedModel, self).__init__()
        self.vit_b_16 = TorchVisionViTModel().get_eager_model()

    def forward(self, input1):
        nchw_input1 = input1.permute(0, 3, 1, 2)
        output = self.vit_b_16(nchw_input1)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./vit_b_16",
        default="./vit_b_16",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "vit_b_16_mtk"
    instance = NhwcWrappedModel()

    # if dropout.p = 0, change probability to 1e-6 to prevent -inf when quantize
    for _name, module in instance.named_modules():
        if isinstance(module, torch.nn.Dropout):
            if module.p == 0:
                module.p = 1e-6

    inputs = (torch.randn(1, 224, 224, 3),)
    build_executorch_binary(
        instance.eval(),
        (torch.randn(1, 224, 224, 3),),
        f"{args.artifact}/{pte_filename}",
        [inputs],
        quant_dtype=Precision.A8W8,
        skip_op_name={
            "aten_permute_copy_default_4",
            "aten_permute_copy_default_18",
            "aten_permute_copy_default_32",
            "aten_permute_copy_default_46",
            "aten_permute_copy_default_60",
            "aten_permute_copy_default_74",
            "aten_permute_copy_default_88",
            "aten_permute_copy_default_102",
            "aten_permute_copy_default_116",
            "aten_permute_copy_default_130",
            "aten_permute_copy_default_144",
            "aten_permute_copy_default_158",
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
