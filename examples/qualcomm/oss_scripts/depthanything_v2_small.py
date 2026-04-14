# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
from multiprocessing.connection import Client

import numpy as np
import requests
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_backend_type,
    get_imagenet_dataset,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from PIL import Image
from torchao.quantization.utils import compute_error
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.modeling_outputs import DepthEstimatorOutput

HUGGING_FACE_DEPTHANYTHING_V2 = "depth-anything/Depth-Anything-V2-Small-hf"


def postprocess_output_and_save(output, image_height, image_width, output_image_path):
    image_processor = AutoImageProcessor.from_pretrained(HUGGING_FACE_DEPTHANYTHING_V2)

    post_processed_output = image_processor.post_process_depth_estimation(
        # Resize the output back to the original image dimensions and set the channel dimension to 1 as
        # depth‑estimation outputs are single‑channel.
        DepthEstimatorOutput(
            predicted_depth=output.reshape(1, image_height, image_width)
        ),
        target_sizes=[(image_height, image_width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    depth.save(output_image_path)


def main(args):
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)
    os.makedirs(args.artifact, exist_ok=True)

    model = AutoModelForDepthEstimation.from_pretrained(
        HUGGING_FACE_DEPTHANYTHING_V2
    ).eval()

    data_num = 100
    if args.ci:
        data_num = 1
        inputs = [(torch.rand(1, 3, 256, 256),)]
        logging.warning(
            "This option is for CI to verify the export flow. It uses random input and will result in poor accuracy."
        )
    elif args.dump_example_output:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(os.path.join(args.artifact, "source.png"))
        image_processor = AutoImageProcessor.from_pretrained(
            HUGGING_FACE_DEPTHANYTHING_V2
        )

        pixel_values = image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        inputs = [(pixel_values,)]
        data_num = 1
    else:
        inputs, _ = get_imagenet_dataset(
            dataset_path=f"{args.dataset}",
            data_size=data_num,
            image_shape=(256, 256),
        )

    goldens = []
    with torch.no_grad():
        for per_input in inputs:
            predicted_depth = model(*per_input).predicted_depth
            goldens.append(predicted_depth.flatten())

    pte_filename = "depthanything_v2_small_qnn"
    # Skip lowering/compilation if using pre-generated PTE
    if not args.pre_gen_pte:
        # Lower to QNN
        backend = get_backend_type(args.backend)
        quant_dtype = {
            QnnExecuTorchBackendType.kGpuBackend: None,
            QnnExecuTorchBackendType.kHtpBackend: QuantDtype.use_8a8w,
        }[backend]
        build_executorch_binary(
            model,
            inputs[0],
            args.model,
            os.path.join(args.artifact, pte_filename),
            inputs,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quant_dtype=quant_dtype,
            backend=backend,
            shared_buffer=args.shared_buffer,
            online_prepare=args.online_prepare,
        )

    if args.compile_only:
        return

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_filename}"
    pte_path = (
        f"{args.pre_gen_pte}/{pte_filename}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
    )

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=workspace,
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        target=args.target,
    )
    adb.push(inputs=inputs, backends={backend})
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(host_output_path=args.artifact)

    evaluations = {
        "sqnr": [],
    }
    for i in range(data_num):
        prediction = torch.from_numpy(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
        )
        evaluations["sqnr"].append(compute_error(goldens[i], prediction))

    if args.dump_example_output:
        example_input_shape = list(inputs[0][0].shape)
        image_height, image_width = example_input_shape[-2], example_input_shape[-1]

        # Post-process source model output and export the depth estimation image
        postprocess_output_and_save(
            goldens[0],
            image_height,
            image_width,
            os.path.join(args.artifact, "golden_depth.png"),
        )
        prediction = np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
        # Post-process QNN output and export the depth estimation image
        postprocess_output_and_save(
            torch.from_numpy(prediction),
            image_height,
            image_width,
            os.path.join(args.artifact, "prediction_depth.png"),
        )

    evaluations["sqnr"] = sum(evaluations["sqnr"]) / data_num
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"sqnr": evaluations["sqnr"]}))
    else:
        print("SQNR(dB)={sqnr}".format(**evaluations))


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./depthanything_v2_small",
        default="./depthanything_v2_small",
        type=str,
    )
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
        "--dump_example_output",
        help=(
            "If specified, export the example image and post-process both the source model output "
            "and the QNN output into depth-estimation images."
        ),
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    args.validate(args)

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
