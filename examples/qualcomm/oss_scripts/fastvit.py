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

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserverWithLossEvaluation,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    MovingAverageMinMaxObserver,
    QuantizationConfig,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
from executorch.examples.qualcomm.utils import (
    get_imagenet_dataset,
    make_output_dir,
    topk_accuracy,
)
from torchao.quantization.pt2e.quantizer import QuantizationSpec


def get_instance(repo_path: str, checkpoint_path: str):
    import sys

    sys.path.insert(0, repo_path)

    from models.modules.mobileone import reparameterize_model
    from timm.models import create_model

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = create_model("fastvit_s12")
    model = reparameterize_model(model).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    data_num = 100
    inputs, targets = get_imagenet_dataset(
        dataset_path=f"{args.dataset}",
        data_size=data_num,
        image_shape=(256, 256),
    )

    pte_filename = "fastvit_qnn"

    def get_custom_quantizer():
        quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_8a8w,
            backend=qnn_config.backend,
            soc_model=qnn_config.soc_model,
        )

        # there are lots of outliers appearing in fastvit parameters
        # we need to apply special configuration to saturate their impact
        act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            qscheme=torch.per_tensor_affine,
            observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(
                **{"averaging_constant": 0.01}
            ),
        )
        weight_qspec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=torch.iinfo(torch.int8).min + 1,
            quant_max=torch.iinfo(torch.int8).max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=PerChannelParamObserverWithLossEvaluation.with_args(
                **{"steps": 100, "use_mse": True}
            ),
        )
        # rewrite default per-channel ptq config
        quantizer.default_quant_config.per_channel_quant_config = QuantizationConfig(
            input_activation=act_qspec,
            output_activation=act_qspec,
            weight=weight_qspec,
            bias=_derived_bias_quant_spec,
        )

        # rewrite default ptq config
        q_config = quantizer.default_quant_config.quant_config
        quantizer.default_quant_config.quant_config = QuantizationConfig(
            input_activation=act_qspec,
            output_activation=act_qspec,
            weight=q_config.weight,
            bias=q_config.bias,
        )
        return quantizer

    # lower to QNN
    quantizer = {
        QnnExecuTorchBackendType.kGpuBackend: None,
        QnnExecuTorchBackendType.kHtpBackend: get_custom_quantizer(),
    }[qnn_config.backend]
    build_executorch_binary(
        model=convert_linear_to_conv2d(
            get_instance(args.oss_repo, args.pretrained_weight)
        ),
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        custom_quantizer=quantizer,
    )

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
            conn.send(json.dumps({f"top_{k}": topk[i] for i, k in enumerate(k_val)}))
    else:
        for i, k in enumerate(k_val):
            print(f"top_{k}->{topk[i]}%")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./fastvit",
        default="./fastvit",
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
        required=True,
    )

    parser.add_argument(
        "--oss_repo",
        help="Path to cloned https://github.com/apple/ml-fastvit",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help=(
            "Location of model pretrained weight."
            "e.g., -p ./fastvit_s12_reparam.pth.tar"
            "Pretrained model can be found in "
            "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_s12_reparam.pth.tar"
        ),
        type=str,
        required=True,
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
