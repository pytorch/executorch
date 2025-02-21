# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import argparse
import json
import os
import random
import time
from multiprocessing.connection import Client
import requests
import io
import torchaudio

import numpy as np

import sphn
import torch

import torch.nn as nn
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

# from executorch.examples.models.llama.llama_transformer import Transformer

# from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from huggingface_hub import hf_hub_download
from moshi.models import loaders

from torch.profiler import profile, ProfilerActivity


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


def mimi_encode():
    None
def mimi_decode():
    None
def mimi_test(mimi, args, max_duration_sec=10.0):
    pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_rate = mimi.sample_rate
    url = "https://huggingface.co/lmz/moshi-swift/resolve/main/bria-24khz.mp3"
    sample_pcm, sample_sr = read_mp3_from_url(url)
    pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_rate = mimi.sample_rate
    sample_pcm = torch.tensor(sample_pcm, device='cpu')
    max_duration_len = int(sample_rate * max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    sample_pcm = sample_pcm[None].to(device='cpu')
    # sample_pcm = torch.ones(1, 1, 240000)

    print("streaming encoding...")
    start_time = time.time()
    all_codes = []

    def run_loop():
        for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
            end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
            chunk = sample_pcm[..., start_idx:end_idx]
            codes = mimi.encode(chunk)
            if codes.shape[-1]:
                print(start_idx, codes.shape, end="\r")
                all_codes.append(codes)
    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_loop()
        prof.export_chrome_trace("trace.json")
    else:
        run_loop()
    all_codes_th = torch.cat(all_codes, dim=-1)
    print(f"codes {all_codes_th.shape} generated in {time.time() - start_time:.2f}s")
    print("streaming decoding...")
    all_pcms = []
    with mimi.streaming(1):
        for i in range(all_codes_th.shape[-1]):
            codes = all_codes_th[..., i : i + 1]
            pcm = mimi.decode(codes)
            print(i, pcm.shape, end="\r")
            all_pcms.append(pcm)
    all_pcms = torch.cat(all_pcms, dim=-1)
    pcm_ref = mimi.decode(all_codes_th)  # same as mimi_decode(input[0])
    print("pcm", all_pcms.shape, all_pcms.dtype)

    assert torch.allclose(pcm_ref, all_pcms, atol=1e-5)

    class MimiDecode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.decode(x)

    mimi_decode = MimiDecode(mimi)

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)
    pte_filename = "mimi_qnn"
    input = (all_codes_th.to(torch.int32),)
    build_executorch_binary(
        mimi_decode.eval(),
        input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        [input],
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
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
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=[input], input_list="input_0_0.raw")
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    # top-k analysis
    predictions = []
    predictions.append(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    )
    htp_res = torch.from_numpy(predictions[0]).view(1, 1, 240000)
    cosine_sim = torch.nn.functional.cosine_similarity(
        pcm_ref.flatten(), htp_res.flatten(), dim=0
    ).item()
    print("Cos similarity: ", cosine_sim)
    sphn.write_wav("streaming_out.wav", all_pcms[0, 0].cpu().numpy(), sample_rate)
    sphn.write_wav("ref.wav", pcm_ref[0, 0].cpu().numpy(), sample_rate)
    sphn.write_wav("htp.wav", htp_res[0,0].cpu().numpy(), sample_rate)
    # With QNN 2.28.2
    # 0.9650231003761292
    # 8a8w    cos similarity: 0.9635128378868103, 1 inference: 73.5ms,    file size: ~59mb
    # 16a16w  cos similarity: failed at runner: Error from rpc transport, file size: ~104mb
    # 16a4w   cos similarity: failed at runner: Error from rpc transport, file size: ~53mb
    # fp      cos similarity: failed at runner: Error from rpc transport, file size: ~101mb (QNN 2.31.0 for this)

    class MimiEncode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.encode(x)

    mimi_encode = MimiEncode(mimi)
    chunk = sample_pcm[..., 0:pcm_chunk_size]
    out = mimi_encode(chunk)
    exported_encode = torch.export.export(mimi_encode, (chunk,), strict=False).module()


def main(args):
    seed_all(42424242)

    print("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, "cpu")
    print("mimi loaded")
    # emb = torch.load('emb.pt')

    with torch.no_grad():
        mimi_test(mimi, args)

if __name__ == "__main__":

    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./ssd300_vgg16",
        default="./mimi",
        type=str,
    )

    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    # parser.add_argument(
    #     "--device", type=str, default="cpu" if torch.cuda.device_count() else "cpu"
    # )
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
