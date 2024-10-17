# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.llama3_2_vision.preprocess.export_preprocess_lib import (
    export_preprocess,
    get_example_inputs,
    lower_to_executorch_preprocess,
)


def main():

    # ExecuTorch
    ep_et = export_preprocess()
    et = lower_to_executorch_preprocess(ep_et)
    with open("preprocess_et.pte", "wb") as file:
        et.write_to_file(file)

    # AOTInductor
    ep_aoti = export_preprocess()
    torch._inductor.aot_compile(
        ep_aoti.module(),
        get_example_inputs(),
        options={"aot_inductor.output_path": "preprocess_aoti.so"},
    )


if __name__ == "__main__":
    main()
