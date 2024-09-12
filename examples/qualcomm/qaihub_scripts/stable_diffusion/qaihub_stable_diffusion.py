# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import numpy as np
import piq
import torch
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)

from executorch.backends.qualcomm.utils.utils import (
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)

from executorch.examples.qualcomm.qaihub_scripts.stable_diffusion.stable_diffusion_lib import (
    StableDiffusion,
)
from executorch.examples.qualcomm.qaihub_scripts.utils.utils import (
    gen_pte_from_ctx_bin,
    get_encoding,
)
from executorch.examples.qualcomm.utils import (
    setup_common_args_and_variables,
    SimpleADB,
)
from PIL import Image
from torchvision.transforms import ToTensor

target_names = ("text_encoder", "unet", "vae")


def get_quant_data(
    encoding: dict, data: torch.Tensor, input_model: str, input_index: int
):
    scale = encoding[f"{input_model}_input"]["scale"][input_index]
    offset = encoding[f"{input_model}_input"]["offset"][input_index]
    if offset < 0:
        quant_data = data.div(scale).sub(offset).clip(min=0, max=65535).detach()
    else:
        quant_data = data.div(scale).add(offset).clip(min=0, max=65535).detach()

    return quant_data.to(dtype=torch.uint16)


def get_encodings(
    path_to_shard_encoder: str,
    path_to_shard_unet: str,
    path_to_shard_vae: str,
    compiler_specs,
):
    text_encoder_encoding = get_encoding(
        path_to_shard=path_to_shard_encoder,
        compiler_specs=compiler_specs,
        get_input=False,
        get_output=True,
        num_input=1,
        num_output=1,
    )
    unet_encoding = get_encoding(
        path_to_shard=path_to_shard_unet,
        compiler_specs=compiler_specs,
        get_input=True,
        get_output=True,
        num_input=3,
        num_output=1,
    )
    vae_encoding = get_encoding(
        path_to_shard=path_to_shard_vae,
        compiler_specs=compiler_specs,
        get_input=True,
        get_output=True,
        num_input=1,
        num_output=1,
    )

    return (
        text_encoder_encoding[0],
        unet_encoding[0],
        unet_encoding[1],
        vae_encoding[0],
        vae_encoding[1],
    )


def get_time_embedding(timestep, time_embedding):
    timestep = torch.tensor([timestep])
    t_emb = get_timestep_embedding(timestep, 320, True, 0)
    emb = time_embedding(t_emb)

    return emb


def build_args_parser():
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="Path for storing generated artifacts by this example. Default ./stable_diffusion_qai_hub",
        default="./stable_diffusion_qai_hub",
        type=str,
    )

    parser.add_argument(
        "--pte_prefix",
        help="Prefix of pte files name. Default qaihub_stable_diffusion",
        default="qaihub_stable_diffusion",
        type=str,
    )

    parser.add_argument(
        "--text_encoder_bin",
        type=str,
        default=None,
        help="[For AI hub ctx binary] Path to Text Encoder.",
        required=True,
    )

    parser.add_argument(
        "--unet_bin",
        type=str,
        default=None,
        help="[For AI hub ctx binary] Path to UNet.",
        required=True,
    )

    parser.add_argument(
        "--vae_bin",
        type=str,
        default=None,
        help="[For AI hub ctx binary] Path to Vae Decoder.",
        required=True,
    )

    parser.add_argument(
        "--prompt",
        default="a photo of an astronaut riding a horse on mars",
        type=str,
        help="Prompt to generate image from.",
    )

    parser.add_argument(
        "--num_time_steps",
        default=20,
        type=int,
        help="The number of diffusion time steps.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Strength of guidance (higher means more influence from prompt).",
    )

    parser.add_argument(
        "--vocab_json",
        type=str,
        help="Path to tokenizer vocab.json file. Can get vocab.json under https://huggingface.co/openai/clip-vit-base-patch32/tree/main",
        required=True,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="folder path to pre-compiled ptes",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--fix_latents",
        help="Enable this option to fix the latents in the unet diffuse step.",
        action="store_true",
    )

    return parser


def broadcast_ut_result(output_image, seed):
    sd = StableDiffusion(seed)
    to_tensor = ToTensor()
    target = sd(args.prompt, 512, 512, args.num_time_steps)
    target = to_tensor(target).unsqueeze(0)
    output_tensor = to_tensor(
        Image.fromarray(np.round(output_image[0] * 255).astype(np.uint8)[0])
    ).unsqueeze(0)

    psnr_piq = piq.psnr(target, output_tensor)
    ssim_piq = piq.ssim(target, output_tensor)
    print(f"PSNR: {round(psnr_piq.item(), 3)}, SSIM: {round(ssim_piq.item(), 3)}")
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"PSNR": psnr_piq.item(), "SSIM": ssim_piq.item()}))


def save_result(output_image):
    img = Image.fromarray(np.round(output_image[0] * 255).astype(np.uint8)[0])
    save_path = f"{args.artifact}/outputs/output_image.jpg"
    img.save(save_path)
    print(f"Output image saved at {save_path}")


def inference(args, compiler_specs, pte_files):
    # Loading a pretrained EulerDiscreteScheduler from the https://huggingface.co/stabilityai/stable-diffusion-2-1-base.
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", revision="main"
    )

    #  Loading a pretrained UNet2DConditionModel (which includes the time embedding) from the https://huggingface.co/stabilityai/stable-diffusion-2-1-base.
    time_embedding = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="unet", revision="main"
    ).time_embedding

    scheduler.set_timesteps(args.num_time_steps)
    scheduler.config.prediction_type = "epsilon"
    # Get encoding of unet and vae
    (
        encoder_output,
        unet_input,
        unet_output,
        vae_input,
        vae_output,
    ) = get_encodings(
        args.text_encoder_bin,
        args.unet_bin,
        args.vae_bin,
        compiler_specs,
    )
    encoding = {
        "encoder_output": encoder_output,
        "unet_input": unet_input,
        "unet_output": unet_output,
        "vae_input": vae_input,
        "vae_output": vae_output,
    }

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=args.build_folder,
        pte_path=pte_files,
        workspace=f"/data/local/tmp/executorch/{args.pte_prefix}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        runner="examples/qualcomm/qaihub_scripts/stable_diffusion/qaihub_stable_diffusion_runner",
    )

    input_unet = ()
    input_list_unet = ""

    for i, t in enumerate(scheduler.timesteps):
        time_emb = get_quant_data(
            encoding, get_time_embedding(t, time_embedding), "unet", 1
        )
        input_list_unet += f"input_{i}_0.raw\n"
        input_unet = input_unet + (time_emb,)

    qnn_executor_runner_args = [
        f"--text_encoder_path {adb.workspace}/{args.pte_prefix}_text_encoder.pte",
        f"--unet_path {adb.workspace}/{args.pte_prefix}_unet.pte",
        f"--vae_path {adb.workspace}/{args.pte_prefix}_vae.pte",
        f"--input_list_path {adb.workspace}/input_list.txt",
        f"--output_folder_path {adb.output_folder}",
        f'--prompt "{args.prompt}"',
        f"--guidance_scale {args.guidance_scale}",
        f"--num_time_steps {args.num_time_steps}",
        f"--vocab_json {adb.workspace}/vocab.json",
    ]
    if args.fix_latents:
        qnn_executor_runner_args.append("--fix_latents")

    text_encoder_output_scale = encoding["encoder_output"]["scale"][0]
    text_encoder_output_offset = encoding["encoder_output"]["offset"][0]
    unet_input_latent_scale = encoding["unet_input"]["scale"][0]
    unet_input_latent_offset = encoding["unet_input"]["offset"][0]
    unet_input_text_emb_scale = encoding["unet_input"]["scale"][2]
    unet_input_text_emb_offset = encoding["unet_input"]["offset"][2]
    unet_output_scale = encoding["unet_output"]["scale"][0]
    unet_output_offset = encoding["unet_output"]["offset"][0]
    vae_input_scale = encoding["vae_input"]["scale"][0]
    vae_input_offset = encoding["vae_input"]["offset"][0]
    vae_output_scale = encoding["vae_output"]["scale"][0]
    vae_output_offset = encoding["vae_output"]["offset"][0]

    qnn_executor_runner_args = qnn_executor_runner_args + [
        f"--text_encoder_output_scale {text_encoder_output_scale}",
        f"--text_encoder_output_offset {text_encoder_output_offset}",
        f"--unet_input_latent_scale {unet_input_latent_scale}",
        f"--unet_input_latent_offset {unet_input_latent_offset}",
        f"--unet_input_text_emb_scale {unet_input_text_emb_scale}",
        f"--unet_input_text_emb_offset {unet_input_text_emb_offset}",
        f"--unet_output_scale {unet_output_scale}",
        f"--unet_output_offset {unet_output_offset}",
        f"--vae_input_scale {vae_input_scale}",
        f"--vae_input_offset {vae_input_offset}",
        f"--vae_output_scale {vae_output_scale}",
        f"--vae_output_offset {vae_output_offset}",
    ]

    qnn_executor_runner_args = " ".join(
        [
            f"cd {adb.workspace} &&",
            f"./qaihub_stable_diffusion_runner {' '.join(qnn_executor_runner_args)}",
        ]
    )

    files = [args.vocab_json]

    if args.fix_latents:
        seed = 42
        latents = torch.randn((1, 4, 64, 64), generator=torch.manual_seed(seed)).to(
            "cpu"
        )
        # We need to explicitly permute after init tensor or else the random value will be different
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents = latents * scheduler.init_noise_sigma
        flattened_tensor = latents.view(-1)
        # Save the flattened tensor to a .raw file
        with open(os.path.join(args.artifact, "latents.raw"), "wb") as file:
            file.write(flattened_tensor.numpy().tobytes())
        files.append(os.path.join(args.artifact, "latents.raw"))

    if not args.skip_push:
        adb.push(inputs=input_unet, input_list=input_list_unet, files=files)
    adb.execute(custom_runner_cmd=qnn_executor_runner_args)

    output_image = []

    def post_process_vae():
        with open(f"{args.artifact}/outputs/output_0_0.raw", "rb") as f:
            output_image.append(
                np.fromfile(f, dtype=np.float32).reshape(1, 512, 512, 3)
            )

    adb.pull(output_path=args.artifact, callback=post_process_vae)

    if args.fix_latents:
        broadcast_ut_result(output_image, seed)
    else:
        save_result(output_image)


def main(args):
    os.makedirs(args.artifact, exist_ok=True)

    # common part for compile & inference
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
        use_multi_contexts=True,
    )
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.model),
        backend_options=backend_options,
        is_from_context_binary=True,
    )

    if args.pre_gen_pte is None:
        # Create custom operators as context loader
        bundle_programs = [
            from_context_binary(args.text_encoder_bin, "ctx_loader_0"),
            from_context_binary(args.unet_bin, "ctx_loader_1"),
            from_context_binary(args.vae_bin, "ctx_loader_2"),
        ]
        pte_names = [f"{args.pte_prefix}_{target_name}" for target_name in target_names]
        pte_files = gen_pte_from_ctx_bin(
            args.artifact, pte_names, compiler_specs, bundle_programs
        )
        assert (
            len(pte_files) == 3
        ), f"Error: Expected 3 PTE files, but got {len(pte_files)} files."

    else:
        pte_files = [
            f"{args.pre_gen_pte}/{args.pte_prefix}_{target_name}.pte"
            for target_name in target_names
        ]
    if args.compile_only:
        return

    inference(args, compiler_specs, pte_files)


if __name__ == "__main__":  # noqa: C901
    parser = build_args_parser()
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
