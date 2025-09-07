# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
import struct
import warnings

import torch

from aot_utils.llm_utils.sanity_checks import (
    check_all_chunks_same_num_layer,
    check_between_inclusive,
    check_exist,
    check_ext,
    check_old_arg,
    check_shapes,
    check_supported_model,
    check_supported_tokenizer,
    check_tokenizer_exist,
    check_weights_exist,
)
from aot_utils.llm_utils.utils import (
    dump_embedding_lut_for_cmdline,
    generate_mask,
    get_dest_path,
    get_dirname,
    get_embedding_layer,
    get_exp_name,
    get_export_shapes,
    get_master_pos_emb,
    get_normalized_config,
    load_checkpoints,
    resolve_model_classes,
)
from aot_utils.mllm_utils.preprocessor_whisper import WhisperAudioProcessor
from datasets import Audio, Dataset, load_dataset
from executorch import exir
from executorch.backends.mediatek import (
    NeuropilotPartitioner,
    NeuropilotQuantizer,
    Precision,
)
from executorch.exir.backend.backend_details import CompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from tqdm import tqdm


warnings.filterwarnings("ignore")


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run Export to ET for suppoorted LLM models.", allow_abbrev=False
    )
    parser.add_argument(
        "config",
        type=str,
        help="[Required] Model config json file. "
        "Model config must be in same directory as all model weight bins and tokenizer files.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="A16W8",
        choices=["A16W4", "A16W8", "A16W16", "A8W4", "A8W8"],
        help="Precision to quantize entire model to.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="DX4",
        choices=["DX3", "DX4"],
        help="Chip model of the inference device. "
        "DX3 for Dimensity 9300, DX4 for Dimensity 9400.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset name or path to dataset. Defaults to None to use random inputs",
    )
    parser.add_argument(
        "-n",
        "--num_chunks",
        type=int,
        default=4,
        help="Number of chunks to cut the model into. Defaults to 4.",
    )
    parser.add_argument(
        "-r",
        "--response_cap",
        type=int,
        default=9,
        help="Max Number of Response Tokens to save during calibration. Defaults to 9.",
    )
    parser.add_argument(
        "--preformatter",
        type=str,
        default=None,
        help="Preformatter Template to use to wrap input with. Defaults to None.",
    )
    parser.add_argument(
        "-shapes",
        nargs="+",
        help="[Required] Expected input shapes to reconfigure TFLites to. Space separated list of "
        "shapes in the format: xtyc (e.g. 32t512c)",
    )

    return parser


# flake8: noqa: F405
def args_sanity_checks(args):
    check_old_arg(args.config)
    check_exist(args.config, "Config file")
    check_ext(args.config, ".json", "Config file")
    config = get_normalized_config(args.config)

    weight_dir = get_dirname(args.config)
    check_tokenizer_exist(weight_dir)
    check_weights_exist(weight_dir)

    check_supported_model(config)
    check_supported_tokenizer(config)

    if args.preformatter is not None:
        check_exist(args.preformatter, "Preformatter json file")
        check_ext(args.preformatter, ".json", "preformatter")

    if args.dataset is not None:
        check_exist(args.dataset)

    check_between_inclusive(
        args.num_chunks, 1, config.llm.num_hidden_layers, "num_chunks"
    )

    check_shapes(args.shapes)


def print_args(args, exp_name):
    print("Please check if all arguments are correct:")
    print(f"Config file:                  {args.config}")
    print(f"Output pte folder:            pte/{exp_name}")
    print(f"Quantization precision:       {args.precision}")
    print(f"Preformatter:                 {args.preformatter}")
    print(f"Calibration Dataset:          {args.dataset}")
    print(f"Max Response Tokens:          {args.response_cap}")
    print(f"Number of chunks:             {args.num_chunks}")
    print(f"Export shape(s):              {args.shapes}")
    print(f"Platform:                     {args.platform}")
    print()


def apply_preformatter(inp, preformatter=None):
    formatted_text = preformatter.generate_prompt(inp["text"])
    inp["text"] = formatted_text
    print(f"Formatted Prompt:\n{formatted_text}")
    return inp


def tokenize_dataset(inp, tokenizer):
    text = inp["text"]
    inp_encoded = tokenizer(text, return_tensors="pt")  # dict
    inp_encoded.pop("attention_mask")
    inp_encoded = inp_encoded["input_ids"]
    inp_encoded = inp_encoded.to(torch.int32)
    inp["input_ids"] = inp_encoded
    inp.pop("text")
    return inp


def process_audio(inp, preprocessor, encoder):
    audio_path = inp["text"]
    if audio_path.endswith("mp3"):
        print(f"Loading audio from path: {audio_path}")
        mm_input = Dataset.from_dict({"audio": [str(audio_path)]}).cast_column(
            "audio", Audio(sampling_rate=16000)
        )[0]["audio"]["array"]
    else:
        print(f"Expected mm input filepath to end with 'mp3', but got {audio_path}")

    processed, _ = preprocessor.preprocess(
        mm_input, sampling_rate=16000, return_tensors="pt"
    )
    inputs = processed["input_features"].to(torch.float32)
    inp["encoder_inputs"] = inputs
    _, cross_cache = encoder(inputs)

    inp["encoded_audio"] = processed
    inp["cross_cache"] = cross_cache

    return inp


def prepare_encoder_inputs(inp):
    inp["hidden_states"] = inp["encoder_inputs"]
    inp.pop("encoder_inputs")
    return inp


def add_instruction(inp, tokenizer):
    prompt = ""
    inp_encoded = tokenizer(prompt, return_tensors="pt")
    inp_encoded.pop("attention_mask")
    inp_encoded = inp_encoded["input_ids"]
    inp_encoded = inp_encoded.to(torch.int32)
    inp["input_ids"] = inp_encoded
    return inp


def reset_cache(
    num_chunks, num_key_value_heads, num_blocks_per_chunk, head_dim, max_cache_size
):
    cache = []
    for i in range(num_chunks):
        curr_chunk_cache = torch.zeros(
            (
                2 * num_blocks_per_chunk[i],
                num_key_value_heads,
                max_cache_size,  # generate fixed cache as torch dynamic shape cannot handle 2 dynamic dim
                head_dim,
            ),
            dtype=torch.float32,
        )
        cache.append(curr_chunk_cache)
    return cache


def forward_and_save(
    models,
    hidden_state,
    cache,
    cross_cache,
    mask,
    pos_emb,
    model_input_dict,
    num_blocks_per_chunk,
    batch_name,
):
    for chunk_idx in range(len(models)):
        cache_in = cache[chunk_idx]
        num_layers = int(cross_cache.shape[0] / 2)
        num_blocks = num_blocks_per_chunk[chunk_idx]
        cross_key = cross_cache[
            chunk_idx * num_blocks : (chunk_idx + 1) * num_blocks, ...
        ]
        cross_value = cross_cache[
            chunk_idx * num_blocks
            + num_layers : (chunk_idx + 1) * num_blocks
            + num_layers,
            ...,
        ]
        cross_cache_in = torch.cat([cross_key, cross_value], dim=0)

        try:
            model_input_dict[str(chunk_idx)] = {
                **model_input_dict[str(chunk_idx)],
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                    "cross_cache": cross_cache_in,
                },
            }
        except:
            model_input_dict[str(chunk_idx)] = {
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                    "cross_cache": cross_cache_in,
                }
            }
        with torch.no_grad():
            model_out = models[chunk_idx](
                hidden_state,
                mask,
                pos_emb,
                cross_cache_in,
                *torch.split(cache_in, 1, dim=0),
            )
        hidden_state = model_out[0]
        cache[chunk_idx] = torch.cat(
            model_out[1 : 1 + 2 * num_blocks_per_chunk[chunk_idx]], dim=0
        ).clone()
    return hidden_state, cache


def prepare_model_inputs(
    inp,
    models,
    embedding_layer,
    master_pos_emb,
    num_blocks_per_chunk,
    num_key_value_heads,
    head_dim,
    max_cache_size,
    eos_token_id_tensor,
    response_cap,
):
    model_input_dict = {str(i): None for i in range(len(models))}
    input_ids = inp.pop("input_ids")
    hidden_state = embedding_layer(torch.tensor(input_ids))
    input_length = hidden_state.shape[1]
    # Assume fixed cache size
    mask = generate_mask(max_cache_size, 0, input_length, input_length)
    pos_emb = master_pos_emb[:, :input_length, :]
    cross_cache = inp.pop("cross_cache")
    cross_cache = torch.tensor(cross_cache)
    # cache shape: num chunks of 2*num_block, num kv heads, c, head dim
    cache = reset_cache(
        len(models), num_key_value_heads, num_blocks_per_chunk, head_dim, max_cache_size
    )

    logits, cache = forward_and_save(
        models,
        hidden_state,
        cache,
        cross_cache,
        mask,
        pos_emb,
        model_input_dict,
        num_blocks_per_chunk,
        "prompt",
    )
    next_token_logits = logits[:, -1, :]  # last layer logits
    next_token = torch.argmax(next_token_logits, dim=-1)

    response_count = 0
    seq_length = input_length
    while True:
        curr_input_id = next_token[:, None].to(torch.int32)
        input_length = curr_input_id.shape[1]
        hidden_state = embedding_layer(curr_input_id)
        mask = generate_mask(max_cache_size, seq_length, input_length, input_length)
        pos_emb = master_pos_emb[:, seq_length : seq_length + input_length, :]

        logits, cache = forward_and_save(
            models,
            hidden_state,
            cache,
            cross_cache,
            mask,
            pos_emb,
            model_input_dict,
            num_blocks_per_chunk,
            f"response{response_count}",
        )
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == eos_token_id_tensor:
            print(f"Found EOS on batch: {response_count}")
            break

        response_count += 1
        seq_length += input_length
        if response_count == response_cap:
            break

    return model_input_dict


def calibrate_model(model, cal_dataset, chunk_idx: str):
    with torch.no_grad():
        for inp in tqdm(cal_dataset, desc="Calibrating Model: "):
            # pass prompt and response
            for batch in tqdm(inp[chunk_idx].keys(), desc="Batch: "):
                if inp[chunk_idx][batch] is not None:
                    inputs_embeds = torch.tensor(inp[chunk_idx][batch]["hidden_state"])
                    mask = torch.tensor(inp[chunk_idx][batch]["mask"])
                    pos_emb = torch.tensor(inp[chunk_idx][batch]["pos_emb"])
                    cache = torch.tensor(inp[chunk_idx][batch]["cache"])
                    cross_cache = torch.tensor(inp[chunk_idx][batch]["cross_cache"])
                    model(
                        inputs_embeds,
                        mask,
                        pos_emb,
                        cross_cache,
                        *torch.split(cache, 1, dim=0),
                    )


def export_to_et_ir(
    output_folder,
    exp_name,
    model,
    precision,
    max_num_token,
    max_cache_size,
    chunk_idx,
    export_shapes,
    platform_b,
    cal_dataset=None,
):
    print(f"Exporting Chunk {chunk_idx} to PTE")
    example_inputs, dynamic_shapes = model.get_example_inputs(
        max_num_token, max_cache_size, True
    )
    print("Getting pre autograd ATen Dialect Graph")
    pre_autograd_aten_dialect = torch.export.export_for_training(
        model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()  # NOTE: Will be replaced with export
    quantizer = NeuropilotQuantizer()
    quantizer.setup_precision(getattr(Precision, precision))
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    # at this point quant min max are inf
    if cal_dataset is not None:
        calibrate_model(prepared_graph, cal_dataset, str(chunk_idx))
    else:
        prepared_graph(*example_inputs)  # dummy calibration
    converted_graph = convert_pt2e(prepared_graph, fold_quantize=False)

    print("Getting ATen Dialect Graph")
    # Fixed Shape Export Here
    for shape, ntok_and_cache in export_shapes.items():
        dest_path = get_dest_path(output_folder, exp_name, shape, chunk_idx)
        print(f"Exporting Shape {shape} to:\n{dest_path}")
        example_inputs = model.get_example_inputs(*ntok_and_cache)
        aten_dialect: exir.ExportedProgram = torch.export.export(
            converted_graph, example_inputs, strict=True
        )

        print("Lowering to Edge Dialect Graph")
        edge_program: exir.EdgeProgramManager = exir.to_edge(
            aten_dialect,
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        del aten_dialect

        print("Delegating Edge Program to Neuropilot Backend")
        compile_spec = [
            CompileSpec("gno", b"LTS"),
            CompileSpec("gno-exp", b""),
            CompileSpec("gno-non-4d-tiling", b""),
            CompileSpec("ImportForever", struct.pack("?", True)),
            CompileSpec("platform-config", platform_b),
        ]
        partitioner = NeuropilotPartitioner(compile_spec)
        delegated_program = edge_program.to_backend(partitioner)
        print("Exported Delegated Program:")
        print(delegated_program.exported_program())
        del edge_program

        print("Transforming delegated program to executorch backend")
        executorch_program = delegated_program.to_executorch(
            config=exir.ExecutorchBackendConfig(
                memory_planning_pass=exir.passes.MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
                extract_delegate_segments=True,
            )
        )

        print(f"ET Model Dest: {dest_path}\n")
        os.makedirs(dest_path.rsplit("/", 1)[0], exist_ok=True)
        with open(dest_path, "wb") as file:
            file.write(executorch_program.buffer)


def export_encoder_to_et_ir(
    enc_output_folder,
    enc_exp_name,
    model,
    precision,
    num_mel_bins,
    platform_b,
    cal_dataset=None,
):
    print(f"Exporting Encoder to PTE")
    example_inputs = model.get_example_inputs(num_mel_bins)
    print("Getting pre autograd ATen Dialect Graph")
    pre_autograd_aten_dialect = torch.export.export_for_training(
        model, example_inputs, strict=True
    ).module()  # NOTE: Will be replaced with export
    quantizer = NeuropilotQuantizer()
    quantizer.setup_precision(getattr(Precision, precision))
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    # at this point quant min max are inf
    if cal_dataset is not None:
        with torch.no_grad():
            for inp in tqdm(cal_dataset, desc="Calibrating Model: "):
                prepared_graph(torch.tensor(inp["hidden_states"]))
    else:
        prepared_graph(*example_inputs)  # dummy calibration
    converted_graph = convert_pt2e(prepared_graph, fold_quantize=False)

    print("Getting ATen Dialect Graph")
    # Fixed Shape Export Here

    file_name = f"{enc_exp_name}.pte"
    dest_path = os.path.join(enc_output_folder, file_name)
    print(f"Exporting encoder to:\n{dest_path}")
    example_inputs = model.get_example_inputs(num_mel_bins)
    aten_dialect: exir.ExportedProgram = torch.export.export(
        converted_graph, example_inputs, strict=True
    )

    print("Lowering to Edge Dialect Graph")
    edge_program: exir.EdgeProgramManager = exir.to_edge(
        aten_dialect,
        compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
    )
    del aten_dialect

    print("Delegating Edge Program to Neuropilot Backend")
    compile_spec = [
        CompileSpec("gno", b"LTS"),
        CompileSpec("gno-exp", b""),
        CompileSpec("gno-non-4d-tiling", b""),
        CompileSpec("ImportForever", struct.pack("?", True)),
        CompileSpec("platform-config", platform_b),
    ]
    partitioner = NeuropilotPartitioner(compile_spec)
    delegated_program = edge_program.to_backend(partitioner)
    print("Exported Delegated Program:")
    print(delegated_program.exported_program())
    del edge_program

    print("Transforming delegated program to executorch backend")
    executorch_program = delegated_program.to_executorch(
        config=exir.ExecutorchBackendConfig(
            memory_planning_pass=exir.passes.MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
    )

    print(f"ET Model Dest: {dest_path}\n")
    os.makedirs(dest_path.rsplit("/", 1)[0], exist_ok=True)
    with open(dest_path, "wb") as file:
        file.write(executorch_program.buffer)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args_sanity_checks(args)
    if args.dataset is None:
        exp_name = f"{get_exp_name(args.config)}_{args.precision}_dummy_cal_{args.num_chunks}_chunks"
    else:
        exp_name = (
            f"{get_exp_name(args.config)}_{args.precision}_{args.num_chunks}_chunks"
        )
    if args.platform == "DX4":
        platform_b = b"mt6991"
    elif args.platform == "DX3":
        platform_b = b"mt6989"
    else:
        raise ValueError(
            f"Platform should be either DX3 or DX4, but got {args.platform}"
        )
    print_args(args, exp_name)
    enc_exp_name = f"{get_exp_name(args.config)}_{args.precision}_encoder"

    config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(
        args.config
    )
    encoder_class = chunk_class[0]
    decoder_class = chunk_class[1]
    tokenizer = tokenizer_class.from_pretrained(
        weight_dir, language="english", task="transcribe"
    )

    preprocessor_attr = config.p.__dict__
    preprocessor = WhisperAudioProcessor(**preprocessor_attr)

    head_dim = config.llm.head_dim

    # Evenly distribute the layers across chunks.
    num_blocks_per_chunk = [
        (config.llm.num_hidden_layers // args.num_chunks)
        + (i < (config.llm.num_hidden_layers % args.num_chunks))
        for i in range(args.num_chunks)
    ]
    check_all_chunks_same_num_layer(num_blocks_per_chunk)  # noqa: F405

    output_folder = os.path.join("pte", exp_name)
    enc_output_folder = os.path.join("pte", enc_exp_name)

    # Load all collected checkpoint files into one giant state_dict
    state_dict = load_checkpoints(weight_dir)

    dump_embedding_lut_for_cmdline(weight_dir, state_dict, config.llm)

    export_shapes, max_num_token, max_cache_size = get_export_shapes(args.shapes)
    print(f"export shapes: {export_shapes}")
    print(f"Max Num Token: {max_num_token}")
    print(f"Max Cache Size: {max_cache_size}")

    embedding_layer = get_embedding_layer(config.llm, weight_dir, state_dict)

    # Instantiate model chunks
    print("Instantiating submodels")

    encoder = encoder_class(config.encoder)
    encoder = encoder.load_weights(state_dict)

    models = []
    for chunk_idx, num_blocks in enumerate(num_blocks_per_chunk):
        decoder_chunk = decoder_class(
            config.llm,
            num_blocks,
            chunk_idx=chunk_idx,
            dtype=torch.float32,
            include_tail=(chunk_idx == args.num_chunks - 1),
            jit_trace=True,
        )
        decoder_chunk = decoder_chunk.load_weights(
            state_dict, sum(num_blocks_per_chunk[:chunk_idx])
        )
        models.append(decoder_chunk)

    cal_dataset = None
    if args.dataset is not None:
        cal_dataset = load_dataset("text", data_files=args.dataset, split="train")
        master_pos_emb = get_master_pos_emb(config.llm, weight_dir, dtype=torch.float32)
        cal_dataset = cal_dataset.map(
            process_audio,
            fn_kwargs={
                "preprocessor": preprocessor,
                "encoder": encoder,
            },
        )

        encoder_cal_dataset = cal_dataset.map(prepare_encoder_inputs)

        cal_dataset = cal_dataset.map(  # keys: encoded_audio, encoder_inputs, cross_cache, input_ids
            add_instruction, fn_kwargs={"tokenizer": tokenizer}
        )

        print("Preparing Model Calibration Inputs...")
        cal_dataset = cal_dataset.map(
            prepare_model_inputs,
            fn_kwargs={
                "models": models,
                "embedding_layer": embedding_layer,
                "master_pos_emb": master_pos_emb,
                "num_blocks_per_chunk": num_blocks_per_chunk,
                "num_key_value_heads": config.llm.num_key_value_heads,
                "head_dim": head_dim,
                "max_cache_size": max_cache_size,
                "eos_token_id_tensor": torch.tensor(tokenizer.eos_token_id),
                "response_cap": args.response_cap,
            },
        )

    export_encoder_to_et_ir(
        enc_output_folder,
        enc_exp_name,
        encoder,
        args.precision,
        config.encoder.num_mel_bins,
        platform_b,
        encoder_cal_dataset,
    )

    for chunk_idx, chunk in enumerate(models):
        export_to_et_ir(
            output_folder,
            exp_name,
            chunk,
            args.precision,
            max_num_token,
            max_cache_size,
            chunk_idx,
            export_shapes,
            platform_b,
            cal_dataset,
        )


if __name__ == "__main__":
    main()
