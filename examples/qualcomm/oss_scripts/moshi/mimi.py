# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import logging
import os
from multiprocessing.connection import Client

import moshi

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
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.oss_scripts.moshi.model.static_mimi import (
    _transformer_kwargs,
    DEFAULT_REPO,
    get_static_mimi,
    MIMI_NAME,
)
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

from huggingface_hub import hf_hub_download
from moshi.models import loaders

from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

MOSHI_VERSION = "0.2.3"


def read_mp3_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure request is successful

    # Convert to a file-like object
    audio_stream = io.BytesIO(response.content)

    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(audio_stream, format="mp3")

    return waveform.numpy(), sample_rate


def compute_scores(cpu_decode_res: torch.Tensor, qnn_decode_res: torch.Tensor):
    assert cpu_decode_res.shape == qnn_decode_res.shape, "Tensor shapes do not match"
    abs_diff = torch.abs(cpu_decode_res - qnn_decode_res)
    atol = torch.max(abs_diff)
    logging.info("Atol: {:.3f}".format(atol))

    cpu_decode_res = cpu_decode_res.float()
    qnn_decode_res = qnn_decode_res.float()
    error = cpu_decode_res - qnn_decode_res
    original_power = torch.mean(torch.pow(cpu_decode_res, 2))
    error_power = torch.mean(torch.pow(error, 2))
    sqnr = 10 * torch.log10(original_power / error_power)
    logging.info("SQNR: {:.3f}".format(sqnr))


def init_inputs():
    num_layers = _transformer_kwargs["num_layers"]
    batch_size = 1  # 1 chunk per batch
    num_heads = _transformer_kwargs["num_heads"]
    context = _transformer_kwargs["context"]
    head_dim = _transformer_kwargs["d_model"] // _transformer_kwargs["num_heads"]

    k_cache = torch.zeros(
        (num_layers, batch_size, num_heads, context, head_dim),
        device="cpu",
        dtype=torch.float32,
    )
    v_cache = torch.zeros(
        (num_layers, batch_size, num_heads, context, head_dim),
        device="cpu",
        dtype=torch.float32,
    )

    # end_index will perform end_index % context
    # end_offset will keep increment even after end_offset > context
    end_index = torch.zeros((num_layers, 1), device="cpu", dtype=torch.long)
    end_offset = torch.zeros((num_layers, 1), device="cpu", dtype=torch.long)

    # partial for transpose conv layers, please refer to StaticRawStreamingConvTranspose1d under static_convtr.py for more info
    # There are total of 5 transpose conv in mimi decoder, 1 is under to_encoder_framerate, and 4 is under SeanetDecoder
    partial_convtr_0 = torch.zeros((1, 512, 2), device="cpu", dtype=torch.float32)
    partial_convtr_1 = torch.zeros((1, 512, 8), device="cpu", dtype=torch.float32)
    partial_convtr_2 = torch.zeros((1, 256, 6), device="cpu", dtype=torch.float32)
    partial_convtr_3 = torch.zeros((1, 128, 5), device="cpu", dtype=torch.float32)
    partial_convtr_4 = torch.zeros((1, 64, 4), device="cpu", dtype=torch.float32)

    # Some index for naming are skipped on purpose as those conv_layers have empty previous
    previous_conv_0 = torch.zeros((1, 512, 6), device="cpu", dtype=torch.float32)
    previous_conv_1 = torch.zeros((1, 512, 2), device="cpu", dtype=torch.float32)
    previous_conv_3 = torch.zeros((1, 256, 2), device="cpu", dtype=torch.float32)
    previous_conv_5 = torch.zeros((1, 128, 2), device="cpu", dtype=torch.float32)
    previous_conv_7 = torch.zeros((1, 64, 2), device="cpu", dtype=torch.float32)
    previous_conv_9 = torch.zeros((1, 64, 2), device="cpu", dtype=torch.float32)

    return (
        k_cache,
        v_cache,
        end_index,
        end_offset,
        partial_convtr_0,
        partial_convtr_1,
        partial_convtr_2,
        partial_convtr_3,
        partial_convtr_4,
        previous_conv_0,
        previous_conv_1,
        previous_conv_3,
        previous_conv_5,
        previous_conv_7,
        previous_conv_9,
    )


def compile_mimi_encoder(
    args,
    orig_mimi,
    encoder_inputs,
    skip_node_id_set,
    skip_node_op_set,
    encoder_pte_filename,
):
    class MimiEncode(nn.Module):
        def __init__(self, mimi: nn.Module):
            super().__init__()
            self.mimi_model = mimi

        def forward(self, x):
            return self.mimi_model.encode(x)

    mimi_encoder_model = MimiEncode(orig_mimi)
    build_executorch_binary(
        mimi_encoder_model.eval(),
        encoder_inputs[0],
        args.model,
        f"{args.artifact}/{encoder_pte_filename}",
        encoder_inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )


def inference_mimi_encoder(args, encoder_inputs, encoder_pte_filename):
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{encoder_pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{encoder_pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=encoder_inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    encoder_predictions = []
    for i in range(len(encoder_inputs)):
        np_arr = np.fromfile(
            os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.int64
        )
        encoder_predictions.append(torch.from_numpy(np_arr).view(1, 8, 1))
    return encoder_predictions


def export_mimi_encoder(
    args, orig_mimi, sample_pcm, pcm_chunk_size, skip_node_id_set, skip_node_op_set
):
    encoder_inputs = []
    count = 0
    cpu_encoded_results = []
    logging.info("streaming encoding...")
    for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
        end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
        chunk = sample_pcm[..., start_idx:end_idx]
        # Preparing QNN inputs
        encoder_inputs.append((chunk,))
        count += 1
        # Performing cpu encoding for golden
        codes = orig_mimi.encode(chunk)
        if codes.shape[-1]:
            cpu_encoded_results.append(codes)

    encoder_pte_filename = "mimi_encoder_qnn"
    if args.use_cpu_encoder:
        logging.info("Using CPU Encoder, Skip Compile and Inference for QNN Encoder")
    elif args.compile_only:
        logging.info("Compile only for QNN Encoder")
        compile_mimi_encoder(
            args,
            orig_mimi,
            encoder_inputs,
            skip_node_id_set,
            skip_node_op_set,
            encoder_pte_filename,
        )
    elif args.pre_gen_pte:
        logging.info("Inference only for QNN Encoder")
        qnn_encoded_results = inference_mimi_encoder(
            args,
            encoder_inputs,
            encoder_pte_filename,
        )
    else:
        logging.info("Compile and Inference for QNN Encoder")
        compile_mimi_encoder(
            args,
            orig_mimi,
            encoder_inputs,
            skip_node_id_set,
            skip_node_op_set,
            encoder_pte_filename,
        )
        qnn_encoded_results = inference_mimi_encoder(
            args,
            encoder_inputs,
            encoder_pte_filename,
        )

    encoded_results = (
        cpu_encoded_results
        if (args.use_cpu_encoder or args.compile_only)
        else qnn_encoded_results
    )

    # These 2 returned values will be same if use cpu_encoder instead of QNN encoder.
    return encoded_results, cpu_encoded_results


def compile_static_mimi_decoder(
    args,
    static_mimi_decoder,
    encoded_results,
    skip_node_id_set,
    skip_node_op_set,
    static_decoder_pte_filename,
):
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )
    quantizer.add_custom_quant_annotations((annotate_mimi_decoder,))

    static_states = init_inputs()

    with torch.no_grad():
        static_mimi_decoder(encoded_results[0], *static_states)

        fx_graph_module = torch.export.export(
            static_mimi_decoder,
            (
                encoded_results[0],
                *static_states,
            ),
            strict=False,
        ).module()

    annotated_model = prepare_pt2e(fx_graph_module, quantizer)
    logging.info("Quantizing the model...")
    for codes in encoded_results:
        _out, *static_states = annotated_model(codes, *static_states)
    quantized_model = convert_pt2e(annotated_model)

    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.model],
        backend_options=backend_options,
    )

    edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
        quantized_model,
        (
            encoded_results[0],
            *static_states,
        ),
        compiler_spec,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
    )

    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ),
    )
    exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
    with open(f"{args.artifact}/{static_decoder_pte_filename}.pte", "wb") as file:
        exec_prog_mgr.write_to_file(file)


def inference_static_mimi_decoder(
    args,
    encoded_results,
    encoded_results_list,
    pcm_chunk_size,
    static_decoder_pte_filename,
):
    workspace = f"/data/local/tmp/executorch/{static_decoder_pte_filename}"
    pte_path = f"{args.artifact}/{static_decoder_pte_filename}.pte"
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            "./qnn_mimi_decoder_runner",
            f"--model_path {workspace}/{static_decoder_pte_filename}.pte",
            f"--output_folder_path {workspace}/outputs",
        ]
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
        runner="examples/qualcomm/oss_scripts/moshi/qnn_mimi_decoder_runner",
    )
    adb.push(inputs=encoded_results)
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    num_chunks = len(encoded_results)
    shape = num_chunks * pcm_chunk_size
    qnn_decode_res = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    ).view(1, 1, shape)

    return qnn_decode_res


def export_mimi_decoder(
    args,
    static_mimi_decoder,
    encoded_results,
    pcm_chunk_size,
    skip_node_id_set,
    skip_node_op_set,
):
    encoded_results_list = ""
    for index, encoder_result in enumerate(encoded_results):
        encoded_results[index] = encoder_result.to(torch.int32)
        encoded_results_list += f"input_{index}_0.raw\n"

    logging.info("streaming decoding...")
    qnn_decode_res = None
    static_decoder_pte_filename = "static_mimi_decoder_qnn"
    with static_mimi_decoder.streaming(1):
        if args.compile_only:
            logging.info("Compile only for QNN Static Decoder")
            compile_static_mimi_decoder(
                args,
                static_mimi_decoder,
                encoded_results,
                skip_node_id_set,
                skip_node_op_set,
                static_decoder_pte_filename,
            )
        elif args.pre_gen_pte:
            logging.info("Inference only for QNN Static Decoder")
            qnn_decode_res = inference_static_mimi_decoder(
                args,
                encoded_results,
                encoded_results_list,
                pcm_chunk_size,
                static_decoder_pte_filename,
            )
        else:
            logging.info("Compile and Inference for QNN Static Decoder")
            compile_static_mimi_decoder(
                args,
                static_mimi_decoder,
                encoded_results,
                skip_node_id_set,
                skip_node_op_set,
                static_decoder_pte_filename,
            )
            qnn_decode_res = inference_static_mimi_decoder(
                args,
                encoded_results,
                encoded_results_list,
                pcm_chunk_size,
                static_decoder_pte_filename,
            )
    return qnn_decode_res


def main(args):
    assert (
        moshi.__version__ == MOSHI_VERSION
    ), f"Please ensure Moshi version == {MOSHI_VERSION}, current version is {moshi.__version__}"

    if args.compile_only and args.pre_gen_pte:
        exit("Cannot set both compile_only and pre_gen_pte as true")

    logging.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, MIMI_NAME)
    orig_mimi = loaders.get_mimi(args.mimi_weight, "cpu")  # For encoder
    static_mimi = get_static_mimi(args.mimi_weight, "cpu")  # For static decoder
    logging.info("mimi loaded")

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)
    os.makedirs(args.artifact, exist_ok=True)

    sample_rate = orig_mimi.sample_rate
    url = "https://huggingface.co/lmz/moshi-swift/resolve/main/bria-24khz.mp3"
    sample_pcm, sample_sr = read_mp3_from_url(url)
    sample_rate = orig_mimi.sample_rate
    sample_pcm = torch.tensor(sample_pcm, device="cpu")
    max_duration_len = int(sample_rate * args.max_duration_sec)
    if sample_pcm.shape[-1] > max_duration_len:
        sample_pcm = sample_pcm[..., :max_duration_len]
    sample_pcm = sample_pcm[None].to(device="cpu")
    # 1920 chunk_size = 0.08sec
    pcm_chunk_size = int(orig_mimi.sample_rate / orig_mimi.frame_rate)

    qnn_decode_res = None
    with torch.no_grad():
        encoded_results, cpu_encoded_results = export_mimi_encoder(
            args,
            orig_mimi,
            sample_pcm,
            pcm_chunk_size,
            skip_node_id_set,
            skip_node_op_set,
        )
        qnn_decode_res = export_mimi_decoder(
            args,
            static_mimi,
            encoded_results,
            pcm_chunk_size,
            skip_node_id_set,
            skip_node_op_set,
        )

        if args.compile_only:
            exit(f"Finish compile_only and saved to {args.artifact}")

        pcm_ref = orig_mimi.decode(torch.cat(cpu_encoded_results, dim=-1))
        logging.info("PCM ref V.S. QNN streaming mode")
        compute_scores(pcm_ref, qnn_decode_res)

        sphn.write_wav(
            f"{args.artifact}/pcm_ref.wav", pcm_ref[0, 0].cpu().numpy(), sample_rate
        )
        sphn.write_wav(
            f"{args.artifact}/qnn_decode_res.wav",
            qnn_decode_res[0, 0].cpu().numpy(),
            sample_rate,
        )


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
        "--max_duration_sec",
        help="Max duration seconds for the audio to be processed.",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated mimi encoder/decoder in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--use_cpu_encoder",
        help="Enable this flag to perform encoder with cpu.",
        action="store_true",
        default=False,
    )

    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_REPO)

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
