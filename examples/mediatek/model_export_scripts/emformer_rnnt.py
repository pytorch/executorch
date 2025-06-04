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

from aot_utils.oss_utils.utils import build_executorch_binary
from executorch.backends.mediatek import Precision
from executorch.examples.models.emformer_rnnt import (
    EmformerRnntJoinerModel,
    EmformerRnntPredictorModel,
    EmformerRnntTranscriberModel,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./emformer_rnnt",
        default="./emformer_rnnt",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build Transcriber
    print("Build Transcriber")
    transcriber = EmformerRnntTranscriberModel()
    t_model = transcriber.get_eager_model()
    inputs = transcriber.get_example_inputs()
    pte_filename = "emformer_rnnt_t_mtk"
    build_executorch_binary(
        t_model.eval(),
        inputs,
        f"{args.artifact}/{pte_filename}",
        [inputs],
        quant_dtype=Precision.A8W8,
        skip_op_type={
            "aten.where.self",
        },
        skip_op_name={
            "aten_div_tensor_mode",
            "aten_unsqueeze_copy_default",
            "aten_unsqueeze_copy_default_1",
            "aten_unsqueeze_copy_default_2",
            "aten_unsqueeze_copy_default_3",
            "aten_unsqueeze_copy_default_4",
            "aten_unsqueeze_copy_default_5",
            "aten_unsqueeze_copy_default_6",
            "aten_unsqueeze_copy_default_7",
            "aten_unsqueeze_copy_default_8",
            "aten_unsqueeze_copy_default_9",
            "aten_unsqueeze_copy_default_10",
            "aten_unsqueeze_copy_default_11",
            "aten_unsqueeze_copy_default_12",
            "aten_unsqueeze_copy_default_13",
            "aten_unsqueeze_copy_default_14",
            "aten_unsqueeze_copy_default_15",
            "aten_unsqueeze_copy_default_16",
            "aten_unsqueeze_copy_default_17",
            "aten_unsqueeze_copy_default_18",
            "aten_unsqueeze_copy_default_19",
        },
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list_t.txt"
    with open(input_list_file, "w") as f:
        f.write("input_t_0_0.bin input_t_0_1.bin")
        f.flush()
    for idx, data in enumerate(inputs[0]):
        file_name = f"{args.artifact}/input_t_0_{idx}.bin"
        data.detach().numpy().tofile(file_name)
    golden = t_model(inputs[0])
    for idx, data in enumerate(golden):
        file_name = f"{args.artifact}/golden_t_0_{idx}.bin"
        data.detach().numpy().tofile(file_name)

    # build Predictor
    print("Build Predictor")
    predictor = EmformerRnntPredictorModel()
    p_model = predictor.get_eager_model()
    inputs = predictor.get_example_inputs()
    pte_filename = "emformer_rnnt_p_mtk"
    build_executorch_binary(
        p_model.eval(),
        inputs,
        f"{args.artifact}/{pte_filename}",
        [inputs],
        quant_dtype=Precision.A8W8,
        skip_op_name={
            "aten_permute_copy_default",
            "aten_embedding_default",
        },
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list_p.txt"
    with open(input_list_file, "w") as f:
        f.write("input_p_0_0.bin input_p_0_1.bin input_p_0_2.bin")
        f.flush()
    for idx, data in enumerate(inputs[0]):
        file_name = f"{args.artifact}/input_p_0_{idx}.bin"
        try:
            data.detach().numpy().tofile(file_name)
        except:
            pass
    golden = p_model(inputs[0])
    for idx, data in enumerate(golden):
        file_name = f"{args.artifact}/golden_p_0_{idx}.bin"
        try:
            data.detach().numpy().tofile(file_name)
        except:
            pass

    # build Joiner
    print("Build Joiner")
    joiner = EmformerRnntJoinerModel()
    j_model = joiner.get_eager_model()
    inputs = joiner.get_example_inputs()
    pte_filename = "emformer_rnnt_j_mtk"
    build_executorch_binary(
        j_model.eval(),
        inputs,
        f"{args.artifact}/{pte_filename}",
        [inputs],
        quant_dtype=Precision.A8W8,
        skip_op_name={
            "aten_add_tensor",
        },
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list_j.txt"
    with open(input_list_file, "w") as f:
        f.write("input_j_0_0.bin input_j_0_1.bin input_j_0_2.bin input_j_0_3.bin")
        f.flush()
    for idx, data in enumerate(inputs[0]):
        file_name = f"{args.artifact}/input_j_0_{idx}.bin"
        data.detach().numpy().tofile(file_name)
    golden = j_model(inputs[0])
    for idx, data in enumerate(golden):
        file_name = f"{args.artifact}/golden_j_0_{idx}.bin"
        data.detach().numpy().tofile(file_name)
