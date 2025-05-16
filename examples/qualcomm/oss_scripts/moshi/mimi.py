# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
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

from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_mimi_decoder,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from huggingface_hub import hf_hub_download
from moshi.models import loaders

from torch.ao.quantization.observer import MinMaxObserver


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


def compute_scores(cpu_decode_res: torch.Tensor, htp_decode_res: torch.Tensor):
    assert cpu_decode_res.shape == htp_decode_res.shape, "Tensor shapes do not match"
    abs_diff = torch.abs(cpu_decode_res - htp_decode_res)
    atol = torch.max(abs_diff)
    print("Atol: ", atol)

    cpu_decode_res = cpu_decode_res.float()
    htp_decode_res = htp_decode_res.float()
    error = cpu_decode_res - htp_decode_res
    original_power = torch.mean(torch.pow(cpu_decode_res, 2))
    error_power = torch.mean(torch.pow(error, 2))
    sqnr = 10 * torch.log10(original_power / error_power)
    print("SQNR: ", sqnr)


def test_decoder_with_emb_input(mimi, args):
    class MimiDecode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.mimi_model.upsample(x)
            (emb,) = self.mimi_model.decoder_transformer(x)
            emb.transpose(1, 2)
            with self.mimi_model._context_for_encoder_decoder:
                out = self.mimi_model.decoder(emb)
            return out

    emb_input = torch.rand(1, 1, 512, device="cpu")
    mimi_decode = MimiDecode(mimi).eval()
    cpu_res = mimi_decode(emb_input)
    pte_filename = "mimi_decoder_emb_qnn"

    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )
    quantizer.add_custom_quant_annotations((annotate_mimi_decoder,))

    emb_inputs = [(emb_input,)]
    build_executorch_binary(
        mimi_decode,
        emb_inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        emb_inputs,
        custom_quantizer=quantizer,
        quant_dtype=QuantDtype.use_16a8w,
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
    adb.push(inputs=emb_inputs, input_list="input_0_0.raw\n")
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    emb_predictions = []
    for i in range(len(emb_inputs)):
        np_arr = np.fromfile(
            os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
        )
        emb_predictions.append(torch.from_numpy(np_arr).view(1, 1, 1920))
    print("Emb input test results")
    compute_scores(cpu_res, emb_predictions[0])


def mimi_encode(
    mimi,
    encode_inputs,
    encoder_input_list,
    pcm_chunk_size,
    skip_node_id_set,
    skip_node_op_set,
) -> torch.Tensor:
    class MimiEncode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.encode(x)

    mimi_encode_model = MimiEncode(mimi)

    pte_filename = "mimi_encoder_qnn"
    build_executorch_binary(
        mimi_encode_model.eval(),
        encode_inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        encode_inputs,
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
    adb.push(inputs=encode_inputs, input_list=encoder_input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    encoder_predictions = []
    # Num chunks should align with args.chunks_per_batch
    num_chunks = encode_inputs[0][0].shape[-1] // pcm_chunk_size
    for i in range(len(encode_inputs)):
        np_arr = np.fromfile(
            os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.int64
        )
        encoder_predictions.append(torch.from_numpy(np_arr).view(1, 8, num_chunks))
    return encoder_predictions


def mimi_decode(
    mimi, encode_res_list, pcm_chunk_size, skip_node_id_set, skip_node_op_set
) -> torch.Tensor:
    class MimiDecode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.decode(x)

    mimi_decode_model = MimiDecode(mimi)
    decode_inputs, decode_input_list = [], ""
    for index, encoder_res in enumerate(encode_res_list):
        decode_inputs.append((encoder_res.to(torch.int32),))
        decode_input_list += f"input_{index}_0.raw\n"

    pte_filename = "mimi_decoder_qnn"

    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )
    quantizer.add_custom_quant_annotations((annotate_mimi_decoder,))

    build_executorch_binary(
        mimi_decode_model.eval(),
        decode_inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        decode_inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        custom_quantizer=quantizer,
        quant_dtype=QuantDtype.use_16a8w,
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
    adb.push(inputs=decode_inputs, input_list=decode_input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    decoder_predictions = []
    # Num chunks should align with args.chunks_per_batch
    num_chunks = decode_inputs[0][0].shape[-1]
    shape = num_chunks * pcm_chunk_size
    for i in range(len(decode_inputs)):
        np_arr = np.fromfile(
            os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
        )
        decoder_predictions.append(torch.from_numpy(np_arr).view(1, 1, shape))
    htp_decode_res = torch.cat(decoder_predictions, dim=-1)

    return htp_decode_res


def export_mimi(mimi, args, max_duration_sec=10.0):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)
    os.makedirs(args.artifact, exist_ok=True)

    if args.emb_input_test:
        test_decoder_with_emb_input(mimi, args)
        return

    sample_rate = mimi.sample_rate
    url = "https://huggingface.co/lmz/moshi-swift/resolve/main/bria-24khz.mp3"
    sample_pcm, sample_sr = read_mp3_from_url(url)
    sample_rate = mimi.sample_rate
    sample_pcm = torch.tensor(sample_pcm, device="cpu")
    max_duration_len = int(sample_rate * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    sample_pcm = sample_pcm[None].to(device="cpu")

    encoder_inputs, encoder_input_list = [], ""
    # 1920 chunk_size = 0.08sec
    pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
    batch_size = pcm_chunk_size * args.chunks_per_batch
    count = 0
    for start_idx in range(0, sample_pcm.shape[-1], batch_size):
        end_idx = min(sample_pcm.shape[-1], start_idx + batch_size)
        chunk = sample_pcm[..., start_idx:end_idx]
        encoder_inputs.append((chunk,))
        encoder_input_list += f"input_{count}_0.raw\n"
        count += 1

    print("streaming encoding...")
    cpu_encode_res = mimi.encode(sample_pcm)
    htp_encode_res = mimi_encode(
        mimi,
        encoder_inputs,
        encoder_input_list,
        pcm_chunk_size,
        skip_node_id_set,
        skip_node_op_set,
    )

    # Leave it here for now, uncomment this to check htp_encoder with cpu_decoder
    # htp_res = torch.cat(htp_encode_res, dim=-1)
    # cpu_decode_htp_encode =  mimi.decode(htp_res)
    # sphn.write_wav("cpu_decode_htp_encode.wav", cpu_decode_htp_encode[0, 0].cpu().numpy(), sample_rate)

    print("streaming decoding...")
    cpu_decode_res = mimi.decode(cpu_encode_res)
    # TODO: Enable streaming mode, which is the correct way to execute 1 chunk at a time.
    # with mimi.streaming(1):
    htp_decode_res = mimi_decode(
        mimi, htp_encode_res, pcm_chunk_size, skip_node_id_set, skip_node_op_set
    )
    compute_scores(cpu_decode_res, htp_decode_res)

    sphn.write_wav(
        f"{args.artifact}/cpu_decode_res.wav",
        cpu_decode_res[0, 0].cpu().numpy(),
        sample_rate,
    )
    sphn.write_wav(
        f"{args.artifact}/htp_decode_res.wav",
        htp_decode_res[0, 0].cpu().numpy(),
        sample_rate,
    )


def main(args):
    seed_all(42424242)

    print("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, "cpu")
    print("mimi loaded")

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

    parser.add_argument(
        "--chunks_per_batch",
        help="Number of chunks to process per time. Default is 1 chunk per batch, which equals to 0.08 second",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--emb_input_test",
        help="This is just a metrics used to compute accuracy scores, not recommended for general users.",
        action="store_true",
        default=False,
    )

    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
