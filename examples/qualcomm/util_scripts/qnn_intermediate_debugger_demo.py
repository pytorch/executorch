# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import numpy as np

import torch

from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_sample import (
    QcomCosineSimilarityComparator,
    QcomMSEComparator,
)
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    OutputFormat,
    QNNIntermediateDebugger,
)

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.inception_v3.model import InceptionV3Model
from executorch.examples.qualcomm.utils import (
    get_imagenet_dataset,
    make_output_dir,
    topk_accuracy,
)

"""QNN Intermediate Debugger Tutorial
In this Python script, we will go through a few key components to enable the
QNN Intermediate Debugger to verify accuracy for each layer within the graph.
"""


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    data_num = 100

    inputs, targets = get_imagenet_dataset(
        dataset_path=f"{args.dataset}",
        data_size=data_num,
        image_shape=(256, 256),
        crop_size=224,
        shuffle=False,
    )
    pte_filename = "ic3_qnn_debug"
    instance = InceptionV3Model()
    source_model = instance.get_eager_model().eval()
    # Init our QNNIntermediateDebugger and pass it in to build_executorch_binary().
    qnn_intermediate_debugger = QNNIntermediateDebugger(sample_input=inputs[0])
    build_executorch_binary(
        model=source_model,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        quant_dtype=QuantDtype.use_8a8w,
        qnn_intermediate_debugger=qnn_intermediate_debugger,
    )

    # We will only perform inference once, for get_imagenet_dataset(), we will set data_num=1.
    data_num = 1
    inputs = [inputs[0]]
    targets = [targets[0]]

    if args.compile_only:
        return

    # Please ensure that dump_intermediate_outputs are set to true when creating SimpleADB
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    # We will pull the debug output and provide them to the Inspector class.
    # We can then provide our own metrics and output type to generate the intermediate debugging results.
    def validate_intermediate_tensor():
        qnn_intermediate_debugger.setup_inspector(
            etdump_path=f"{args.artifact}/etdump.etdp",
            debug_buffer_path=f"{args.artifact}/debug_output.bin",
        )

        edge_result = qnn_intermediate_debugger.edge_ep.module()(
            *(qnn_intermediate_debugger.sample_input)
        )

        # Highly Recommended: Ensures that edge module accuracy aligns with nn.Module
        with torch.no_grad():
            source_result = source_model(*(qnn_intermediate_debugger.sample_input))
            score = torch.nn.functional.cosine_similarity(
                edge_result.flatten(), source_result.flatten(), dim=0
            ).item()
            print("Cosine Similarity Score between nn.Module and Edge CPU is: ", score)
        # Users can generate multiple comparison metrics in a single execution.

        cos_comparator = qnn_intermediate_debugger.create_comparator(
            QcomCosineSimilarityComparator, threshold=0.9
        )
        qnn_intermediate_debugger.generate_results(
            title="ic3_cos_similarity_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPH,
            comparator=cos_comparator,
        )

        qnn_intermediate_debugger.generate_results(
            title="ic3_cos_similarity_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.CSV_FILE,
            comparator=cos_comparator,
        )

        mse_comparator = qnn_intermediate_debugger.create_comparator(
            QcomMSEComparator, threshold=0.1
        )
        qnn_intermediate_debugger.generate_results(
            title="ic3_mse_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPH,
            comparator=mse_comparator,
        )

    adb.pull_debug_output(
        args.artifact, args.artifact, callback=validate_intermediate_tensor
    )

    adb.pull(host_output_path=args.artifact)

    # top-k analysis
    predictions = []
    for i in range(data_num):
        predictions.append(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
        )

    k_val = [1, 5]
    topk = [topk_accuracy(predictions, targets, k).item() for k in k_val]
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "svg_path": f"{args.artifact}/ic3_mse_debugging_graph.svg",
                        "csv_path": f"{args.artifact}/ic3_cos_similarity_debugging_graph.csv",
                    }
                )
            )
    else:
        for i, k in enumerate(k_val):
            print(f"top_{k}->{topk[i]}%")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation folder of ImageNet dataset. "
            "e.g. --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)"
        ),
        type=str,
        required=False,
    )

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./inception_v3_debug",
        default="./inception_v3_debug",
        type=str,
    )

    args = parser.parse_args()
    try:
        assert (
            args.dump_intermediate_outputs
        ), "In order to use intermediate tensor debugger, please provide the flag --dump_intermediate_outputs when executing."
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
