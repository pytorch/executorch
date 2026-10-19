# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import logging
import os

import numpy as np
from executorch.backends.qualcomm.export_utils import (
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    get_imagenet_dataset,
    make_output_dir,
    topk_accuracy,
)
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.exporters import ExecutorchExporter, ExecutorchQnnConfig

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def get_instance():
    module = (
        AutoModelForImageClassification.from_pretrained(
            "facebook/deit-base-distilled-patch16-224"
        )
        .eval()
        .to("cpu")
    )

    return module


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    os.makedirs(args.artifact, exist_ok=True)
    config = AutoConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
    data_num = 100
    height = config.image_size
    width = config.image_size

    inputs, targets = get_imagenet_dataset(
        dataset_path=f"{args.dataset}",
        data_size=data_num,
        image_shape=(height, width),
        crop_size=(height, width),
    )
    calibration_dataset = [{"pixel_values": feature} for (feature,) in inputs]
    # Get the Deit model.
    model = get_instance()

    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_8a8w,
        backend=qnn_config.backend,
        soc_model=qnn_config.soc_model,
    )
    et_program_manager = ExecutorchExporter().export(
        model,
        sample_inputs=calibration_dataset[0],
        config=ExecutorchQnnConfig(
            soc_model=args.soc_model,
            calibration_dataset=calibration_dataset,
            pt2e_quantizer=quantizer,
            backend="qnn",
            backend_hardware=args.backend,
            alloc_graph_input=True,
            alloc_graph_output=True,
        ),
    )
    pte_filename = "deit_qnn"
    pte_path = f"{args.artifact}/{pte_filename}.pte"
    with open(pte_path, "wb") as f:
        et_program_manager.write_to_file(f)
    logging.info(f"wrote {pte_path} ({os.path.getsize(pte_path)} bytes)")

    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=pte_path,
        workspace=f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_filename}",
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

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

    for i, k in enumerate(k_val):
        print(f"top_{k}->{topk[i]}%")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./deit",
        default="./hf_deit",
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

    args = parser.parse_args()
    main(args)
