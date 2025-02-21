# # Copyright (c) Qualcomm Innovation Center, Inc.
# # All rights reserved
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# import json
# import os
# import sys
# from multiprocessing.connection import Client

# import numpy as np

# import torch
# from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
# from executorch.examples.qualcomm.utils import (
#     build_executorch_binary,
#     get_imagenet_dataset,
#     make_output_dir,
#     parse_skip_delegation_node,
#     setup_common_args_and_variables,
#     SimpleADB,
#     topk_accuracy,
# )

# from huggingface_hub import hf_hub_download

# # python examples/qualcomm/oss_scripts/moshi/moshi.py   -b build-android -H mlgtw-linux2 -s acfa9311 -m SM8650


# def main(args):
#     pte_filename = "moshi_qnn"
#     sys.path.insert(0, "../moshi/moshi/")
#     from moshi.models import LMGen, loaders
#     from moshi.modules.seanet import SEANetEncoder

#     skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

#     # ensure the working directory exist.
#     os.makedirs(args.artifact, exist_ok=True)

#     # ---------------------------------method1------------------------
#     # _seanet_kwargs = {
#     #     "channels": 1,
#     #     "dimension": 512,
#     #     "causal": True,
#     #     "n_filters": 64,
#     #     "n_residual_layers": 1,
#     #     "activation": "ELU",
#     #     "compress": 2,
#     #     "dilation_base": 2,
#     #     "disable_norm_outer_blocks": 0,
#     #     "kernel_size": 7,
#     #     "residual_kernel_size": 3,
#     #     "last_kernel_size": 3,
#     #     # We train using weight_norm but then the weights are pre-processed for inference so
#     #     # that we can use a normal convolution.
#     #     "norm": "none",
#     #     "pad_mode": "constant",
#     #     "ratios": [8, 6, 5, 4],
#     #     "true_skip": True,
#     # }

#     # mimi = SEANetEncoder(**_seanet_kwargs).eval()
#     # ---------------------------------method1------------------------

#     mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
#     mimi = loaders.get_mimi(mimi_weight, device="cpu")
#     mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.

#     wav = (torch.randn(1, 1, 24000 * 10),)  # should be [B, C=1, T]

#     build_executorch_binary(
#         mimi.eval(),
#         wav,
#         args.model,
#         f"{args.artifact}/{pte_filename}",
#         [wav],
#         skip_node_id_set=skip_node_id_set,
#         skip_node_op_set=skip_node_op_set,
#         quant_dtype=QuantDtype.use_8a8w,
#         shared_buffer=args.shared_buffer,
#     )
#     # import pdb; pdb.set_trace()
#     # with torch.no_grad():
#     #     codes = mimi.encode(wav)  # [B, K = 8, T]
#     #     decoded = mimi.decode(codes) #################################################### Unused, more like showcase

#     #     # Supports streaming too.
#     #     frame_size = int(mimi.sample_rate / mimi.frame_rate)
#     #     all_codes = []
#     #     with mimi.streaming(batch_size=1):
#     #         for offset in range(0, wav.shape[-1], frame_size):
#     #             frame = wav[:, :, offset: offset + frame_size]
#     #             codes = mimi.encode(frame)
#     #             assert codes.shape[-1] == 1, codes.shape
#     #             all_codes.append(codes)

#     # ## WARNING: When streaming, make sure to always feed a total amount of audio that is a multiple
#     # #           of the frame size (1920), otherwise the last frame will not be complete, and thus
#     # #           will not be encoded. For simplicity, we recommend feeding in audio always in multiple
#     # #           of the frame size, so that you always know how many time steps you get back in `codes`.

#     # # Now if you have a GPU around.
#     # # mimi.cuda()
#     # moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
#     # moshi = loaders.get_moshi_lm(moshi_weight, device='cpu')
#     # lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)  # this handles sampling params etc.
#     # out_wav_chunks = []
#     # # Now we will stream over both Moshi I/O, and decode on the fly with Mimi.
#     # with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
#     #     for idx, code in enumerate(all_codes):
#     #         tokens_out = lm_gen.step(code)
#     #         # tokens_out is [B, 1 + 8, 1], with tokens_out[:, 1] representing the text token.
#     #         if tokens_out is not None:
#     #             wav_chunk = mimi.decode(tokens_out[:, 1:])
#     #             out_wav_chunks.append(wav_chunk)
#     #         print(idx, end='\r')
#     # out_wav = torch.cat(out_wav_chunks, dim=-1)
#     # import pdb; pdb.set_trace()

#     from datasets import Audio, load_dataset
#     from transformers import AutoFeatureExtractor, MimiModel

#     librispeech_dummy = load_dataset(
#         "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
#     )

#     # load model and feature extractor
#     model = MimiModel.from_pretrained("kyutai/mimi")
#     feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

#     # load audio sample
#     librispeech_dummy = librispeech_dummy.cast_column(
#         "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
#     )
#     # audio_sample = lib[-1]["audio"]["array"]
#     audio_sample = torch.randn(24000 * 10)
#     inputs = feature_extractor(
#         raw_audio=audio_sample,
#         sampling_rate=feature_extractor.sampling_rate,
#         return_tensors="pt",
#     )
#     audio_sample = (
#         inputs["input_values"],
#         inputs["padding_mask"],
#     )
#     build_executorch_binary(
#         model.eval(),
#         audio_sample,
#         args.model,
#         f"{args.artifact}/{pte_filename}",
#         [audio_sample],
#         skip_node_id_set=skip_node_id_set,
#         skip_node_op_set=skip_node_op_set,
#         quant_dtype=QuantDtype.use_8a8w,
#         shared_buffer=args.shared_buffer,
#     )
#     import pdb

#     pdb.set_trace()
#     encoder_outputs = model.encode(
#         inputs["input_values"], inputs["padding_mask"]
#     )  # torch.randn(1, 240000), torch.randn(1, 1, 240000)
#     audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
#     # or the equivalent with a forward pass
#     audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values


# if __name__ == "__main__":
#     parser = setup_common_args_and_variables()

#     parser.add_argument(
#         "-a",
#         "--artifact",
#         help="path for storing generated artifacts by this example. Default ./ssd300_vgg16",
#         default="./moshi",
#         type=str,
#     )

#     args = parser.parse_args()
#     try:
#         main(args)
#     except Exception as e:
#         if args.ip and args.port != -1:
#             with Client((args.ip, args.port)) as conn:
#                 conn.send(json.dumps({"Error": str(e)}))
#         else:
#             raise Exception(e)
