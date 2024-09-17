import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
import struct
import warnings

import torch

from aot_utils.llm_utils.preformatter import Preformatter
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
    get_master_rot_emb,
    get_normalized_config,
    load_checkpoints,
    resolve_model_classes,
)
from datasets import load_dataset
from executorch import exir
from executorch.backends.mediatek import (
    NeuropilotPartitioner,
    NeuropilotQuantizer,
    Precision,
)
from executorch.exir.backend.backend_details import CompileSpec
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
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

    check_between_inclusive(args.num_chunks, 1, config.num_hidden_layers, "num_chunks")

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
    mask,
    pos_emb,
    model_input_dict,
    num_blocks_per_chunk,
    batch_name,
):
    for chunk_idx in range(len(models)):
        cache_in = cache[chunk_idx]

        try:
            model_input_dict[str(chunk_idx)] = {
                **model_input_dict[str(chunk_idx)],
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                },
            }
        except:
            model_input_dict[str(chunk_idx)] = {
                batch_name: {
                    "hidden_state": hidden_state,
                    "mask": mask,
                    "pos_emb": pos_emb,
                    "cache": cache_in,
                }
            }
        with torch.no_grad():
            model_out = models[chunk_idx](
                hidden_state, mask, pos_emb, *torch.split(cache_in, 1, dim=0)
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
    master_rot_emb,
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
    pos_emb = master_rot_emb[:, :, :input_length, :]
    # cache shape: num chunks of 2*num_block, num kv heads, c, head dim
    cache = reset_cache(
        len(models), num_key_value_heads, num_blocks_per_chunk, head_dim, max_cache_size
    )  # empty kv
    logits, cache = forward_and_save(
        models,
        hidden_state,
        cache,
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
        pos_emb = master_rot_emb[:, :, seq_length : seq_length + input_length, :]
        logits, cache = forward_and_save(
            models,
            hidden_state,
            cache,
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
                    model(inputs_embeds, mask, pos_emb, *torch.split(cache, 1, dim=0))


def export_to_et_ir(
    output_folder,
    exp_name,
    model,
    precision,
    max_num_token,
    max_cache_size,
    chunk_idx,
    export_shapes,
    cal_dataset=None,
):
    print(f"Exporting Chunk {chunk_idx} to PTE")
    example_inputs, dynamic_shapes = model.get_example_inputs(
        max_num_token, max_cache_size, True
    )
    print("Getting pre autograd ATen Dialect Graph")
    pre_autograd_aten_dialect = torch._export.capture_pre_autograd_graph(
        model, example_inputs, dynamic_shapes=dynamic_shapes
    )  # NOTE: Will be replaced with export
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
            converted_graph, example_inputs
        )

        print("Lowering to Edge Dialect Graph")
        edge_program: exir.EdgeProgramManager = exir.to_edge(
            aten_dialect,
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        del aten_dialect

        print("Delegating Edge Program to Neuropilot Backend")
        compile_spec = [
            CompileSpec("gno", struct.pack("3s", b"LTS")),
            CompileSpec("gno-exp", struct.pack("0s", b"")),
            CompileSpec("gno-non-4d-tiling", struct.pack("0s", b"")),
            CompileSpec("ImportForever", struct.pack("?", True)),
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
    print_args(args, exp_name)

    config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(
        args.config
    )
    tokenizer = tokenizer_class.from_pretrained(weight_dir)
    if args.preformatter is not None:
        preformatter = Preformatter(args.preformatter)

    head_dim = int(config.hidden_size / config.num_attention_heads)

    # Evenly distribute the layers across chunks.
    num_blocks_per_chunk = [
        (config.num_hidden_layers // args.num_chunks)
        + (i < (config.num_hidden_layers % args.num_chunks))
        for i in range(args.num_chunks)
    ]
    check_all_chunks_same_num_layer(num_blocks_per_chunk)  # noqa: F405

    output_folder = os.path.join("pte", exp_name)

    # Load all collected checkpoint files into one giant state_dict
    state_dict = load_checkpoints(weight_dir)

    dump_embedding_lut_for_cmdline(weight_dir, state_dict, config)

    export_shapes, max_num_token, max_cache_size = get_export_shapes(args.shapes)
    print(f"export shapes: {export_shapes}")
    print(f"Max Num Token: {max_num_token}")
    print(f"Max Cache Size: {max_cache_size}")

    # Instantiate model chunks
    print("Instantiating submodels")
    models = []
    for chunk_idx, num_blocks in enumerate(num_blocks_per_chunk):
        chunk = chunk_class(
            config,
            num_blocks,
            chunk_idx=chunk_idx,
            dtype=torch.float32,
            include_tail=(chunk_idx == args.num_chunks - 1),
            jit_trace=True,
        )
        chunk = chunk.load_weights(state_dict, sum(num_blocks_per_chunk[:chunk_idx]))
        models.append(chunk)

    cal_dataset = None
    if args.dataset is not None:
        cal_dataset = load_dataset("text", data_files=args.dataset, split="train")
        embedding_layer = get_embedding_layer(config, weight_dir, state_dict)
        master_rot_emb = get_master_rot_emb(config, dtype=torch.float32)
        if args.preformatter is not None:
            cal_dataset = cal_dataset.map(
                apply_preformatter, fn_kwargs={"preformatter": preformatter}
            )
        cal_dataset = cal_dataset.map(
            tokenize_dataset, fn_kwargs={"tokenizer": tokenizer}
        )
        print("Preparing Model Calibration Inputs...")
        cal_dataset = cal_dataset.map(
            prepare_model_inputs,
            fn_kwargs={
                "models": models,
                "embedding_layer": embedding_layer,
                "master_rot_emb": master_rot_emb,
                "num_blocks_per_chunk": num_blocks_per_chunk,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": head_dim,
                "max_cache_size": max_cache_size,
                "eos_token_id_tensor": torch.tensor(tokenizer.eos_token_id),
                "response_cap": args.response_cap,
            },
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
            cal_dataset,
        )


if __name__ == "__main__":
    main()
