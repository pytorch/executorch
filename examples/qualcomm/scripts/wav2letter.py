# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import sys
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.wav2letter import Wav2LetterModel
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)


class Conv2D(torch.nn.Module):
    def __init__(self, stride, padding, weight, bias=None):
        super().__init__()
        use_bias = bias is not None
        self.conv = torch.nn.Conv2d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=[weight.shape[2], 1],
            stride=[*stride, 1],
            padding=[*padding, 0],
            bias=use_bias,
        )
        self.conv.weight = torch.nn.Parameter(weight.unsqueeze(-1))
        if use_bias:
            self.conv.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        return self.conv(x)


def get_dataset(data_size, artifact_dir):
    from torch.utils.data import DataLoader
    from torchaudio.datasets import LIBRISPEECH

    def collate_fun(batch):
        waves, labels = [], []

        for wave, _, text, *_ in batch:
            waves.append(wave.squeeze(0))
            labels.append(text)
        # need padding here for static ouput shape
        waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        return waves, labels

    dataset = LIBRISPEECH(artifact_dir, url="test-clean", download=True)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=data_size,
        shuffle=True,
        collate_fn=lambda x: collate_fun(x),
    )
    # prepare input data
    inputs, targets = [], []
    for wave, label in data_loader:
        for index in range(data_size):
            # reshape input tensor to NCHW
            inputs.append((wave[index].reshape(1, 1, -1, 1),))
            targets.append(label[index])
        # here we only take first batch, i.e. 'data_size' tensors
        break

    return inputs, targets


def eval_metric(pred, target_str):
    from torchmetrics.text import CharErrorRate, WordErrorRate

    def parse(ids):
        vocab = " abcdefghijklmnopqrstuvwxyz'*"
        return ["".join([vocab[c] for c in id]).replace("*", "").upper() for id in ids]

    pred_str = parse(
        [
            torch.unique_consecutive(pred[i, :, :].argmax(0))
            for i in range(pred.shape[0])
        ]
    )
    wer, cer = WordErrorRate(), CharErrorRate()
    return wer(pred_str, target_str), cer(pred_str, target_str)


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    instance = Wav2LetterModel()
    # target labels " abcdefghijklmnopqrstuvwxyz'*"
    instance.vocab_size = 29
    model = instance.get_eager_model().eval()
    if args.pretrained_weight:
        model.load_state_dict(torch.load(args.pretrained_weight, weights_only=True))
    else:
        logging.warning(
            "It is strongly recommended to provide pretrained weights, otherwise accuracy will be bad. This option is here mainly for CI purpose to ensure compile is successful."
        )

    # convert conv1d to conv2d in nn.Module level will only introduce 2 permute
    # nodes around input & output, which is more quantization friendly.
    for i in range(len(model.acoustic_model)):
        for j in range(len(model.acoustic_model[i])):
            module = model.acoustic_model[i][j]
            if isinstance(module, torch.nn.Conv1d):
                model.acoustic_model[i][j] = Conv2D(
                    stride=module.stride,
                    padding=module.padding,
                    weight=module.weight,
                    bias=module.bias,
                )

    # retrieve dataset, will take some time to download
    data_num = 100
    if args.ci:
        inputs = [(torch.rand(1, 1, 700, 1),)]
        logging.warning(
            "This option is for CI to verify the export flow. It uses random input and will result in poor accuracy."
        )
    else:
        inputs, targets = get_dataset(data_size=data_num, artifact_dir=args.artifact)
    pte_filename = "w2l_qnn"
    build_executorch_binary(
        model,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        sys.exit(0)

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    adb.pull(output_path=args.artifact)

    predictions = []
    for i in range(data_num):
        predictions.append(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
        )

    # evaluate metrics
    wer, cer = 0, 0
    for i, pred in enumerate(predictions):
        pred = torch.from_numpy(pred).reshape(1, instance.vocab_size, -1)
        wer_eval, cer_eval = eval_metric(pred, targets[i])
        wer += wer_eval
        cer += cer_eval

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps({"wer": wer.item() / data_num, "cer": cer.item() / data_num})
            )
    else:
        print(f"wer: {wer / data_num}\ncer: {cer / data_num}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./wav2letter",
        default="./wav2letter",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help=(
            "Location of pretrained weight, please download via "
            "https://github.com/nipponjo/wav2letter-ctc-pytorch/tree/main?tab=readme-ov-file#wav2letter-ctc-pytorch"
            " for torchaudio.models.Wav2Letter version"
        ),
        default=None,
        type=str,
        required=False,
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
