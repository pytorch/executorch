# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import argparse
import io
import json
import os
import random
from multiprocessing.connection import Client

import numpy as np
import requests

import sphn
import torch

import torch.nn as nn
import torchaudio
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from huggingface_hub import hf_hub_download
from moshi.models import loaders

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_mp3_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure request is successful

    # Convert to a file-like object
    audio_stream = io.BytesIO(response.content)

    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(audio_stream, format="mp3")

    return waveform.numpy(), sample_rate


def mimi_encode(mimi, sample_pcm, skip_node_id_set, skip_node_op_set) -> torch.Tensor:
    class MimiEncode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.encode(x)

    mimi_encode = MimiEncode(mimi)
    encode_input = (sample_pcm,)
    pte_filename = "mimi_encoder_qnn"
    build_executorch_binary(
        mimi_encode.eval(),
        encode_input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        [encode_input],
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )

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
    adb.push(inputs=[encode_input], input_list="input_0_0.raw")
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    predictions = []
    predictions.append(
        np.fromfile(os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.int64)
    )
    htp_res = torch.from_numpy(predictions[0]).view(1, 8, 125)
    return htp_res


def mimi_decode(mimi, encode_res, skip_node_id_set, skip_node_op_set) -> torch.Tensor:
    class MimiDecode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.decode(x)

    mimi_decode = MimiDecode(mimi)
    decode_input = (encode_res,)
    pte_filename = "mimi_decoder_qnn"
    build_executorch_binary(
        mimi_decode.eval(),
        decode_input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        [decode_input],
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )

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
    adb.push(inputs=[decode_input], input_list="input_0_0.raw")
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    predictions = []
    predictions.append(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    )
    htp_decode_res = torch.from_numpy(predictions[0]).view(1, 1, 240000)
    return htp_decode_res


def export_mimi(mimi, args, max_duration_sec=10.0):
    sample_rate = mimi.sample_rate
    url = "https://huggingface.co/lmz/moshi-swift/resolve/main/bria-24khz.mp3"
    sample_pcm, sample_sr = read_mp3_from_url(url)
    sample_rate = mimi.sample_rate
    sample_pcm = torch.tensor(sample_pcm, device="cpu")
    max_duration_len = int(sample_rate * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    sample_pcm = sample_pcm[None].to(device="cpu")

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    print("streaming encoding...")
    cpu_encode_res = mimi.encode(sample_pcm)
    htp_encode_res = mimi_encode(mimi, sample_pcm, skip_node_id_set, skip_node_op_set)
    cpu_decode_res = mimi.decode(cpu_encode_res)
    htp_decode_res = mimi_decode(
        mimi, htp_encode_res.to(torch.int32), skip_node_id_set, skip_node_op_set
    )
    sphn.write_wav(
        "cpu_decode_res.wav", cpu_decode_res[0, 0].cpu().numpy(), sample_rate
    )
    sphn.write_wav(
        "htp_decode_res.wav", htp_decode_res[0, 0].cpu().numpy(), sample_rate
    )


def main(args):
    seed_all(42424242)

    print("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, "cpu")
    print("mimi loaded")
    # emb = torch.load('emb.pt')

    with torch.no_grad():
        export_mimi(mimi, args)


if __name__ == "__main__":

    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./mimi",
        default="./mimi",
        type=str,
    )

    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)

    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
