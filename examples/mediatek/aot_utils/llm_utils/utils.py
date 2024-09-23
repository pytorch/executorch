import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import json
import math

import numpy as np
import torch
from safetensors.torch import load_file


# flake8: noqa: C901


def _get_embedding_weight(weight_dir, state_dict):
    try_last = False
    checkpoint_filename = None
    if state_dict is None:
        checkpoint_files = [
            os.path.join(weight_dir, f)
            for f in os.listdir(weight_dir)
            if (
                (f.startswith("pytorch_model") and f.endswith(".bin"))
                or (f.startswith("model") and f.endswith(".safetensors"))
            )
        ]

        for f in checkpoint_files:
            if "pytorch_model.bin" in f:
                checkpoint_filename = f
                break
            elif "pytorch_model-00001-of" in f:
                checkpoint_filename = f
                break
            elif "model.safetensors" in f:
                checkpoint_filename = f
                break
            elif "model-00001-of" in f:
                checkpoint_filename = f
                break
        if checkpoint_filename is None:
            raise FileNotFoundError(
                f"Unable to find the first checkpoint file in {weight_dir}! "
                "This folder must have either the file pytorch_model.bin or "
                "pytorch_model-00001-of-XXXXX.bin or "
                "model.safetensors or "
                "model-00001-of-XXXXX.safetensors."
            )

        if checkpoint_filename.endswith(".bin"):
            state_dict = torch.load(
                checkpoint_filename, map_location="cpu", weights_only=True
            )
        elif checkpoint_filename.endswith(".safetensors"):
            state_dict = load_file(checkpoint_filename, device="cpu")
        try_last = True

    state_dict_keys = list(state_dict.keys())

    expected_embedding_subkey = "embed_tokens.weight"

    embed_key = None
    for key in state_dict_keys:
        if expected_embedding_subkey in key:
            embed_key = key
            break
    if embed_key is None:
        if try_last:
            if (
                checkpoint_filename == "pytorch_model.bin"
                or checkpoint_filename == "model.safetensors"
            ):
                print("state_dict keys:", state_dict_keys)
                raise KeyError(
                    f"Cannot find embedding layer weight inside {checkpoint_filename}. "
                    f"Please ensure embedding layer weight key contains {expected_embedding_subkey}"
                )
            else:
                checkpoint_filename = checkpoint_filename.replace(
                    "00001", checkpoint_filename.split("-")[-1].split(".")[0]
                )
                if checkpoint_filename.endswith(".bin"):
                    state_dict = torch.load(
                        checkpoint_filename, map_location="cpu", weights_only=True
                    )
                elif checkpoint_filename.endswith(".safetensors"):
                    state_dict = load_file(checkpoint_filename, device="cpu")
                state_dict_keys = list(state_dict.keys())
                for key in state_dict_keys:
                    if expected_embedding_subkey in key:
                        embed_key = key
                        break
                if embed_key is None:
                    print("state_dict keys:", state_dict_keys)
                    raise KeyError(
                        f"Cannot find embedding layer weight inside {checkpoint_filename}. "
                        f"Please ensure embedding layer weight key contains {expected_embedding_subkey}"
                    )
        else:
            print("state_dict keys:", state_dict_keys)
            raise KeyError(
                f"Cannot find embedding layer weight inside state dict. "
                f"Please ensure embedding layer weight key contains {expected_embedding_subkey}"
            )
    return state_dict[embed_key]


def chunk_and_tokenize_prompt(
    prompt,
    tokenizer,
    sub_responses,
    max_len,
    response_handler,
    preformatter=None,
    wikitext=False,
):
    if max_len == 0:
        # No chunking
        if preformatter is not None:
            prompt_formatted = preformatter.generate_prompt(prompt, None)
        else:
            prompt_formatted = prompt

        if tokenizer is None:
            with response_handler:
                print("Prompt tokens:")
                print(prompt)
            prompt_tokens = prompt_formatted
        else:
            with response_handler:
                if preformatter is not None:
                    print(f"Prompt (with {preformatter.name} preformatter):")
                    print(prompt)
                else:
                    print("Prompt:")
                    print(prompt)
            prompt_tokens = tokenizer(prompt_formatted, return_tensors="np")[
                "input_ids"
            ].astype(np.int32)
        return prompt_tokens, None
    else:
        if wikitext:
            # Wikitext chunking, tokenized already
            if prompt.shape[1] < max_len:
                return prompt, None
            else:
                return prompt[:, :max_len], prompt[:, max_len:]

        else:
            # Oppo streaming prompt chunking
            sentences = prompt.split("\n")
            chunked = False
            curr_chunk = ""
            prev_chunk = None
            prev_chunk_tokens = None

            for sentence in sentences:
                if preformatter is not None:
                    if len(sub_responses) == 0:
                        curr_chunk_formatted = preformatter.generate_prompt(
                            curr_chunk, None
                        )
                    else:
                        curr_chunk_formatted = preformatter.generate_prompt(
                            curr_chunk, sub_responses[-1]
                        )
                else:
                    curr_chunk_formatted = curr_chunk
                if tokenizer is None:
                    curr_chunk_tokens = curr_chunk_formatted
                else:
                    curr_chunk_tokens = tokenizer(
                        curr_chunk_formatted, return_tensors="np"
                    )["input_ids"].astype(np.int32)

                if curr_chunk_tokens.shape[1] < max_len:
                    prev_chunk = curr_chunk
                    prev_chunk_tokens = curr_chunk_tokens
                    curr_chunk += sentence + "\n"
                else:
                    chunked = True
                    break

            if prev_chunk_tokens is None:
                raise RuntimeError(
                    f"Length of a single line ({curr_chunk_tokens.shape[1]}) is more than maximum length to chunk prompt to ({max_len})"
                )

            if chunked:
                with response_handler:
                    if preformatter is not None:
                        if len(sub_responses) == 0:
                            print(f"Prompt (with {preformatter.name} preformatter):")
                            print(prev_chunk)
                        else:
                            print(
                                f"Prompt (with {preformatter.name} preformatter with input):"
                            )
                            print(prev_chunk)
                    else:
                        print("Prompt:")
                        print(prev_chunk)
                return prev_chunk_tokens, prompt.split(prev_chunk)[1]
            else:
                with response_handler:
                    if preformatter is not None:
                        if len(sub_responses) == 0:
                            print(f"Prompt (with {preformatter.name} preformatter):")
                            print(curr_chunk)
                        else:
                            print(
                                f"Prompt (with {preformatter.name} preformatter with input):"
                            )
                            print(curr_chunk)
                    else:
                        print("Prompt:")
                        print(curr_chunk)
                return curr_chunk_tokens, None


def dump_embedding_lut_for_cmdline(weight_dir, state_dict, config):
    model_name = os.path.basename(weight_dir)
    output_path = os.path.join(weight_dir, f"embedding_{model_name}_fp32.bin")
    if not os.path.exists(output_path):
        embedding = (
            _get_embedding_weight(weight_dir, state_dict)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        with open(output_path, "wb") as f:
            f.write(embedding.flatten().tobytes())
        print(f"cmdline LUT embedding bin exported to {output_path}")


def generate_alibi(
    cache_size,
    valid_cache,
    input_length,
    valid_input,
    num_heads,
    batch_size=1,
    pytorch=False,
):
    assert (
        valid_input <= input_length
    ), "valid_input must be less than or equal to input_length"
    assert (
        valid_cache <= cache_size
    ), "valid_cache must be less than or equal to cache_size"
    valid_seq_length = valid_cache + valid_input
    total_valid = np.ones((batch_size, valid_seq_length), dtype=np.int32)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** ((-(2 ** -(math.log2 - 3))))
    powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
    slopes = np.power(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** ((-(2 ** -(math.log2(2 * closest_power_of_2) - 3))))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=np.int32)
        slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

    arange_tensor = ((np.cumsum(total_valid, axis=-1) - 1))[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    alibi = alibi.reshape(batch_size, num_heads, 1, valid_seq_length)

    pre_pad_length = cache_size - valid_cache
    pre_pad_tensor = np.zeros(
        (batch_size, num_heads, 1, pre_pad_length), dtype=np.float32
    )
    post_pad_length = input_length - valid_input
    post_pad_tensor = np.zeros(
        (batch_size, num_heads, 1, post_pad_length), dtype=np.float32
    )
    alibi = np.concatenate([pre_pad_tensor, alibi, post_pad_tensor], axis=-1).astype(
        np.float32
    )

    if pytorch:
        return torch.from_numpy(alibi.copy())
    return alibi.copy()


def generate_mask(
    cache_size,
    valid_cache,
    input_length,
    valid_input,
    batch_size=1,
    mask_value=-100.0,
    pytorch=True,
):
    assert (
        valid_cache <= cache_size
    ), "valid_cache must be less than or equal to cache_size"
    assert (
        valid_input <= input_length
    ), "valid_input must be less than or equal to input_length"
    # Cache mask portion
    valid = np.zeros((1, 1, 1, valid_cache + input_length), dtype=np.float32)
    cache_mask = np.full(
        (1, 1, 1, cache_size - valid_cache), mask_value, dtype=np.float32
    )
    cache_mask = np.concatenate((cache_mask, valid), axis=-1)
    cache_mask_final_shape = np.broadcast_to(
        cache_mask, (batch_size, 1, input_length, cache_size + input_length)
    )

    # Attention mask portion
    mask_cond = np.arange(valid_input)
    triangle = mask_cond >= (mask_cond + 1).reshape(valid_input, 1)
    small_attention_mask = triangle.astype(np.float32) * mask_value
    attention_mask = np.pad(
        small_attention_mask,
        (0, input_length - valid_input),
        "constant",
        constant_values=mask_value,
    )
    attention_mask_with_cache = np.concatenate(
        [np.zeros((input_length, cache_size), dtype=np.float32), attention_mask],
        axis=-1,
    )
    attention_mask_final_shape = np.broadcast_to(
        attention_mask_with_cache[None, None, :, :],
        (batch_size, 1, input_length, cache_size + input_length),
    )

    combined_mask = attention_mask_final_shape + cache_mask_final_shape

    if pytorch:
        return torch.from_numpy(combined_mask.copy())
    return combined_mask.copy()


def get_dest_path(output_folder, exp_name, shape, chunk_idx):
    dest_folder_root = output_folder + f"_{shape}"
    os.makedirs(dest_folder_root, exist_ok=True)
    fname = f"{exp_name}_{shape}_{chunk_idx}.pte"
    dest_path = os.path.join(dest_folder_root, fname)

    return dest_path


def get_dirname(file_path):
    return os.path.dirname(file_path)


def get_exp_name(config_path):
    weight_dir = get_dirname(config_path)
    weight_name = os.path.basename(weight_dir)
    config_name = os.path.basename(config_path).split(".json")[0].replace("config", "")
    if config_name == "":
        exp_name = f"{weight_name}"
    else:
        if config_name.startswith("_"):
            config_name = config_name[1:]
        exp_name = f"{weight_name}_{config_name}"
    return exp_name


def get_embedding_layer(config, weight_dir, state_dict):
    embedding_weight = _get_embedding_weight(weight_dir, state_dict)

    model = torch.nn.Embedding(config.vocab_size, config.hidden_size, -1)
    embed_state_dict = {}
    embed_state_dict["weight"] = embedding_weight.to(torch.float32)
    model.load_state_dict(embed_state_dict)
    return model


def get_export_shapes(shapes):
    export_shapes = {}
    max_num_token = 0
    max_cache_size = 0
    for shape in shapes:
        print(f"Shape: {shape}")
        num_token = int(shape.split("t")[0])
        cache_size = int(shape.split("t")[1].split("c")[0])
        export_shapes[shape] = [num_token, cache_size]
        max_num_token = num_token if num_token > max_num_token else max_num_token
        max_cache_size = cache_size if cache_size > max_cache_size else max_cache_size

    return export_shapes, max_num_token, max_cache_size


def get_master_rot_emb(config, dtype):
    rot_dim = int(config.hidden_size / config.num_attention_heads)
    length = config.max_position_embeddings

    if config.ntk_scaling_factor != 1.0:
        base = (10000 * config.ntk_scaling_factor) ** (rot_dim / (rot_dim - 2))
    else:
        base = 10000

    inv_freq = 1.0 / (
        base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim)
    )  # (rot_dim/2)
    t = np.arange(length, dtype=np.float32)  # (len)
    freqs = np.einsum("i,j->ij", t, inv_freq)  # (len, rot_dim/2)
    emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
    master_cos = np.cos(emb)[None, None, :, :]  # (1,1,len,rot_dim)
    master_sin = np.sin(emb)[None, None, :, :]  # (1,1,len,rot_dim)

    rot_emb = np.concatenate((master_cos, master_sin), axis=1)

    if isinstance(dtype, torch.dtype):
        return torch.from_numpy(rot_emb).to(dtype)
    else:
        return rot_emb.astype(dtype)


def get_normalized_config(config_filepath):
    config_file = json.load(open(config_filepath, "r"))
    if config_file["model_type"] == "llama":
        from models.llm_models.configuration_llama import LlamaConfig as config_class
    config = config_class(**config_file, verbose=False)
    return config


def get_sorted_path_list(folder, ext=".", absolute=False):
    if absolute:
        sorted_list = sorted(
            os.listdir(folder), key=lambda f: int(f.rsplit("_", 1)[1].split(ext)[0])
        )
        return [os.path.join(folder, x) for x in sorted_list]
    else:
        return sorted(
            os.listdir(folder), key=lambda f: int(f.rsplit("_", 1)[1].split(ext)[0])
        )


def load_checkpoints(weight_dir):
    checkpoint_files = [
        os.path.join(weight_dir, f)
        for f in os.listdir(weight_dir)
        if (f.startswith("pytorch_model") and f.endswith(".bin"))
        or (f.startswith("model") and f.endswith(".safetensors"))
    ]
    if len(checkpoint_files) == 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!No model weight files found! Using fake weights!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if len(checkpoint_files) == 0:
        return None

    state_dict = {}
    print("Loading weights from disk")
    is_safetensors = checkpoint_files[0].endswith(".safetensors")
    for i in range(len(checkpoint_files)):
        if is_safetensors:
            state_dict = {**state_dict, **load_file(checkpoint_files[i], device="cpu")}
        else:
            state_dict = {
                **state_dict,
                **torch.load(
                    checkpoint_files[i], map_location="cpu", weights_only=True
                ),
            }

    return state_dict


def resolve_model_classes(
    config_filepath, bypass_tokenizer=False, response_handler=None
):
    config_file = json.load(open(config_filepath, "r"))
    weight_dir = get_dirname(config_filepath)
    if config_file["model_type"] == "llama":
        from models.llm_models.configuration_llama import LlamaConfig as config_class
        from models.llm_models.modeling_llama import LlamaModelChunk as chunk_class
    config = config_class(**config_file, response_handler=response_handler)
    if bypass_tokenizer:
        return config, weight_dir, chunk_class
    else:
        if config.tokenizer == "default":
            if config_file["model_type"] == "llama":
                from aot_utils.llm_utils.tokenizers_.tokenization_llama import (
                    LlamaTokenizer as tokenizer_class,
                )
        else:
            if config.tokenizer == "llama":
                from aot_utils.llm_utils.tokenizers_.tokenization_llama import (
                    LlamaTokenizer as tokenizer_class,
                )
            elif config.tokenizer == "pretrained_fast":
                from aot_utils.llm_utils.tokenizers_.tokenization_utils_fast import (
                    PreTrainedTokenizerFast as tokenizer_class,
                )
        return config, weight_dir, tokenizer_class, chunk_class
