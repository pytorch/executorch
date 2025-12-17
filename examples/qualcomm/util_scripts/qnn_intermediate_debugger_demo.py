# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client
from typing import Any, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from executorch.backends.qualcomm.debugger.metrics_evaluator import (
    CosineSimilarityEvaluator,
    MetricEvaluatorBase,
)
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    OutputFormat,
    QNNIntermediateDebugger,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.devtools import Inspector
from executorch.examples.models.inception_v3.model import InceptionV3Model
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)

"""QNN Intermediate Debugger Tutorial
In this Python script, we will go through a few key components to enable the
QNN Intermediate Debugger to verify accuracy for each layer within the graph.
"""


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

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
    )
    pte_filename = "ic3_qnn_debug"
    instance = InceptionV3Model()
    source_model = instance.get_eager_model().eval()

    # Init our QNNIntermediateDebugger and pass it in to build_executorch_binary().
    qnn_intermediate_debugger = QNNIntermediateDebugger()
    build_executorch_binary(
        source_model,
        instance.get_example_inputs(),
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
        dump_intermediate_outputs=args.dump_intermediate_outputs,
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
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        dump_intermediate_outputs=args.dump_intermediate_outputs,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    class RootMeanSquaredErrorEvaluator(MetricEvaluatorBase):
        def __init__(self, threshold=0.02):
            self.threshold = threshold

        def metric_name(self) -> str:
            return "Root Mean Squared Error"

        def evaluate(
            self, qnn_output: torch.Tensor, cpu_output: torch.Tensor
        ) -> Tuple[Any, bool]:
            mse = F.mse_loss(qnn_output, cpu_output)
            rmse = torch.sqrt(mse)
            valid = rmse < self.threshold
            return rmse, valid

    # We will pull the debug output and provide them to the Inspector class.
    # We can then provide our own metrics and output type to generate the intermediate debugging results.
    def validate_intermediate_tensor():
        inspector = Inspector(
            etdump_path=f"{args.artifact}/etdump.etdp",
            debug_buffer_path=f"{args.artifact}/debug_output.bin",
        )

        qnn_intermediate_debugger.capture_golden(*(inputs[0]))

        # Optional: Ensures that edge module accuracy aligns with nn.Module
        with torch.no_grad():
            edge_result = qnn_intermediate_debugger.edge_module(*(inputs[0]))[0]
            source_result = source_model(*(inputs[0]))
            score = torch.nn.functional.cosine_similarity(
                edge_result.flatten(), source_result.flatten(), dim=0
            ).item()
            print("Cosine Similarity Score between nn.Module and Edge CPU is: ", score)

        # Users can generate multiple comparison metrics in a single execution.
        # Below, we generate 3 metrics.
        qnn_intermediate_debugger.generate_results(
            title="ic3_cos_similarity_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPHS,
            inspector=inspector,
            evaluator=CosineSimilarityEvaluator(0.9),
        )
        qnn_intermediate_debugger.generate_results(
            title="ic3_cos_similarity_csv",
            path=args.artifact,
            output_format=OutputFormat.CSV_FILES,
            inspector=inspector,
            evaluator=CosineSimilarityEvaluator(0.9),
        )
        # Using self defined metrics to print svg graphs
        qnn_intermediate_debugger.generate_results(
            title="ic3_rmse_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPHS,
            inspector=inspector,
            evaluator=RootMeanSquaredErrorEvaluator(0.9),
        )

    adb.pull_debug_output(
        args.artifact, args.artifact, callback=validate_intermediate_tensor
    )

    adb.pull(output_path=args.artifact)

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
                        "svg_path": f"{args.artifact}/ic3_rmse_debugging_graph.svg",
                        "csv_path": f"{args.artifact}/ic3_cos_similarity_csv.csv",
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
