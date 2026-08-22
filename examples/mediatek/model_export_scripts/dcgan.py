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
import os

import dcgan_main

import torch
from aot_utils.oss_utils.utils import build_executorch_binary
from executorch.backends.mediatek import Precision


class NhwcWrappedModel(torch.nn.Module):
    def __init__(self, is_gen=True):
        super(NhwcWrappedModel, self).__init__()
        if is_gen:
            self.dcgan = dcgan_main.Generator()
        else:
            self.dcgan = dcgan_main.Discriminator()

    def forward(self, input1):
        nchw_input1 = input1.permute(0, 3, 1, 2)
        output = self.dcgan(nchw_input1)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. " "Default ./dcgan",
        default="./dcgan",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # prepare dummy data
    inputG = torch.randn(1, 1, 1, 100)
    inputD = torch.randn(1, 64, 64, 3)

    # build Generator
    netG_instance = NhwcWrappedModel(True)
    netG_pte_filename = "dcgan_netG_mtk"
    build_executorch_binary(
        netG_instance.eval(),
        (torch.randn(1, 1, 1, 100),),
        f"{args.artifact}/{netG_pte_filename}",
        [(inputG,)],
        quant_dtype=Precision.A8W8,
    )

    # build Discriminator
    netD_instance = NhwcWrappedModel(False)
    netD_pte_filename = "dcgan_netD_mtk"
    build_executorch_binary(
        netD_instance.eval(),
        (torch.randn(1, 64, 64, 3),),
        f"{args.artifact}/{netD_pte_filename}",
        [(inputD,)],
        quant_dtype=Precision.A8W8,
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list_G.txt"
    with open(input_list_file, "w") as f:
        f.write("inputG_0_0.bin")
        f.flush()
    file_name = f"{args.artifact}/inputG_0_0.bin"
    inputG.detach().numpy().tofile(file_name)
    file_name = f"{args.artifact}/goldenG_0_0.bin"
    goldenG = netG_instance(inputG)
    goldenG.detach().numpy().tofile(file_name)

    input_list_file = f"{args.artifact}/input_list_D.txt"
    with open(input_list_file, "w") as f:
        f.write("inputD_0_0.bin")
        f.flush()
    file_name = f"{args.artifact}/inputD_0_0.bin"
    inputD.detach().numpy().tofile(file_name)
    file_name = f"{args.artifact}/goldenD_0_0.bin"
    goldenD = netD_instance(inputD)
    goldenD.detach().numpy().tofile(file_name)
