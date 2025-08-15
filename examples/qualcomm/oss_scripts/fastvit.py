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
from executorch.backends.qualcomm._passes.expand_broadcast_tensor_shape import (
    ExpandBroadcastTensorShape,
)

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.quantizer.annotators import (
    QuantizationConfig,
    QuantizationSpec,
)
from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    MovingAverageMinMaxObserver,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)


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
    )

    pte_filename = "fastvit_qnn"
    quantizer = make_quantizer(quant_dtype=QuantDtype.use_8a8w)

    # there are lots of outliers appearing in fastvit parameters
    # we need to apply special configuration to saturate their impact
    act_qspec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(
            **{"averaging_constant": 0.02}
        ),
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=torch.iinfo(torch.int8).min + 1,
        quant_max=torch.iinfo(torch.int8).max,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelParamObserver.with_args(
            **{"steps": 200, "use_mse": True}
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

    # lower to QNN
    passes_job = get_capture_program_passes()
    passes_job[ExpandBroadcastTensorShape][QCOM_PASS_ACTIVATE_KEY] = True
    build_executorch_binary(
        convert_linear_to_conv2d(get_instance(args.oss_repo, args.pretrained_weight)),
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        custom_quantizer=quantizer,
        passes_job=passes_job,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        return

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

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
