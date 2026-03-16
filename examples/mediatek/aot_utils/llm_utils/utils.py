# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    medusa_mask=None,
    sliding_window=False,
    sliding_window_size=None,
    pytorch=True,
    bita_prefix_length=0,
    bita_draft_length=0,
):

    assert (
        valid_cache <= cache_size
    ), "valid_cache must be less than or equal to cache_size"

    if sliding_window:  # SWA
        assert sliding_window_size is not None, "sliding_window_size is None"
        # Full mask: (input_length, cache_size + input_length)
        full_mask_length = cache_size + input_length
        # 1. generate invalid input portion
        invalid_input_part = np.full(
            (batch_size, 1, input_length - valid_input, full_mask_length),
            mask_value,
            dtype=np.float32,
        )

        # 2. valid input + invalid cache part
        invalid_cache_part = np.full(
            (batch_size, 1, valid_input, cache_size - valid_cache),
            mask_value,
            dtype=np.float32,
        )

        # 3. valid input + invalid input part
        valid_input_vs_invalid_input_part = np.full(
            (batch_size, 1, valid_input, input_length - valid_input),
            mask_value,
            dtype=np.float32,
        )

        # 4. valid input + valid_cache, valid_input part
        valid_part = np.full(
            (batch_size, 1, valid_input, valid_cache + valid_input),
            mask_value,
            dtype=np.float32,
        )

        # swap to zeros when within sliding window size
        for i in range(valid_part.shape[2]):
            end_position = valid_cache + i
            start_position = max(0, end_position - sliding_window_size + 1)
            valid_part[:, :, i, start_position : end_position + 1] = 0

        # combine everything together
        final_mask = np.concatenate(
            [invalid_cache_part, valid_part, valid_input_vs_invalid_input_part], axis=-1
        )
        final_mask = np.concatenate([final_mask, invalid_input_part], axis=-2)
        combined_mask = final_mask

    else:  # Normal attention
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
        combined_mask[
            :,
            :,
            -bita_draft_length:,
            -valid_cache
            - input_length
            - bita_prefix_length : -valid_cache
            - input_length,
        ] = 0

    if medusa_mask is not None:
        medusa_len = medusa_mask.size(-1)
        combined_mask[:, :, -medusa_len:, -medusa_len:][medusa_mask == 0] = mask_value

    if pytorch:
        return torch.from_numpy(combined_mask.copy())
    return combined_mask.copy()


def get_dest_path(output_folder, exp_name, shape=None, chunk_idx=0):
    dest_folder_root = output_folder + f"{f'_{shape}' if shape is not None else ''}"
    os.makedirs(dest_folder_root, exist_ok=True)
    fname = f"{exp_name}{f'_{shape}' if shape is not None else ''}_{chunk_idx}.pte"
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


def get_master_pos_emb(config, weight_dir, dtype, **kwargs):
    """Generates the master positional embeddings.

    Args:
        config (object): The configuration object containing model parameters.
        dtype (type): The data type of the embeddings.
        kwargs: Other kwargs.

    Returns:
        np.ndarray or torch.Tensor: The master positional embeddings.
    """

    state_dict = kwargs.pop("state_dict", None)

    if config.model_type == "whisper_decoder":
        weight_dir = weight_dir
        if weight_dir is None and state_dict is None:
            raise ValueError("Expect either weight_dir or state_dict, but got neither")

        expected_path = os.path.join(weight_dir, "embedding_pos_fp16.bin")
        if os.path.exists(expected_path):
            embedding_weight = np.fromfile(expected_path, dtype=np.float16).reshape(
                1, config.max_position_embeddings, config.hidden_size
            )
        else:
            embedding_weight = (
                state_dict["model.decoder.embed_positions.weight"].unsqueeze(0).numpy()
            )
            embedding_weight.astype(np.float16).tofile(expected_path)

        if isinstance(dtype, torch.dtype):
            return torch.tensor(embedding_weight).to(dtype)
        return embedding_weight.astype(dtype)


def get_master_rot_emb(config, dtype, **kwargs):
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
    if partial_rotary_factor is None:
        partial_rotary_factor = 1

    rot_dim = (
        int(config.head_dim * partial_rotary_factor)
        if config.model_type not in ["gemma2", "gemma3", "qwen2", "qwen3"]
        else config.head_dim
    )
    length = int(config.max_position_embeddings * config.ntk_scaling_factor)

    if config.ntk_scaling_factor != 1.0:
        base = (10000 * config.ntk_scaling_factor) ** (rot_dim / (rot_dim - 2))
    else:
        base = 10000

    if getattr(config, "rope_scaling", None) is not None:
        if config.rope_scaling["type"] == "longrope":
            short_factor = config.rope_scaling["short_factor"]
            long_factor = config.rope_scaling["long_factor"]
            original_max_position_embeddings = config.original_max_position_embeddings
            length = original_max_position_embeddings
            assert original_max_position_embeddings is not None, (
                "Using longrope for rope scaling but"
                "original_max_position_embeddings is None."
            )

            ext_factors_long = torch.tensor(long_factor, dtype=torch.float32)
            ext_factors = torch.tensor(short_factor, dtype=torch.float32)

            inv_freq_shape = (
                torch.arange(0, rot_dim, 2, dtype=torch.int64).float() / rot_dim
            )
            inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
            inv_freq = inv_freq.unsqueeze(1)
            t = torch.arange(length, dtype=torch.float32)
            t = t.unsqueeze(0)

            inv_freq_long = 1.0 / (ext_factors_long * base**inv_freq_shape)
            inv_freq_long = inv_freq_long.unsqueeze(1)
            t_long = torch.arange(
                config.max_position_embeddings - length, dtype=torch.float32
            )
            t_long = t_long.unsqueeze(0)

            # Short factor
            freqs = (inv_freq.float() @ t.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = config.max_position_embeddings / original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1 + math.log(scale) / math.log(original_max_position_embeddings)
                )
            master_cos = emb.cos() * scaling_factor
            master_sin = emb.sin() * scaling_factor
            master_cos = master_cos[None, None, :, :]
            master_sin = master_sin[None, None, :, :]

            # Long factor
            freqs_long = (inv_freq_long.float() @ t_long.float()).transpose(0, 1)
            emb_long = torch.cat((freqs_long, freqs_long), dim=-1)

            scale = config.max_position_embeddings / original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1 + math.log(scale) / math.log(original_max_position_embeddings)
                )
            master_cos_long = emb_long.cos() * scaling_factor
            master_sin_long = emb_long.sin() * scaling_factor
            master_cos_long = master_cos_long[None, None, :, :]
            master_sin_long = master_sin_long[None, None, :, :]

            master_cos = torch.cat((master_cos, master_cos_long), dim=2)
            master_sin = torch.cat((master_sin, master_sin_long), dim=2)
            rot_emb = torch.cat((master_cos, master_sin), dim=1)

        elif config.rope_scaling["type"] == "mrope":
            # position_ids = config.e.mrope_position_ids
            # rope_delta = config.e.mrope_delta
            position_ids = kwargs.get("qwen2_vl_position_ids")
            rope_delta = kwargs.get("qwen2_vl_mrope_delta")
            assert (
                position_ids is not None
            ), "Must pass position_ids when using Qwen2-VL mrope."
            assert (
                rope_delta is not None
            ), "Must pass rope_delta when using Qwen2-VL mrope."

            mrope_section = config.rope_scaling["mrope_section"] * 2
            cache_position = torch.arange(
                position_ids.shape[2],
                config.max_position_embeddings,
                dtype=torch.float32,
            )
            delta = cache_position + rope_delta
            delta = delta.view(1, -1).expand(1, -1)
            delta = delta.unsqueeze(0).expand(3, -1, -1)
            position_ids = torch.cat([position_ids, delta], dim=-1)

            inv_freq = 1.0 / (
                base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
            )  # (rot_dim/2)
            inv_freq_expanded = (
                inv_freq[None, None, :, None]
                .float()
                .expand(3, position_ids.shape[1], -1, 1)
            )
            position_ids_expanded = position_ids[
                :, :, None, :
            ].float()  # (3, bs, 1, positions)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            master_cos = emb.cos()
            master_sin = emb.sin()

            cos = torch.cat(
                [
                    m[i % 3]
                    for i, m in enumerate(master_cos.split(mrope_section, dim=-1))
                ],
                dim=-1,
            ).unsqueeze(1)
            sin = torch.cat(
                [
                    m[i % 3]
                    for i, m in enumerate(master_sin.split(mrope_section, dim=-1))
                ],
                dim=-1,
            ).unsqueeze(1)
            rot_emb = torch.cat([cos, sin], dim=1)

        elif config.rope_scaling["type"] == "yarn":
            scaling_factor = config.rope_scaling["factor"]
            extrapolation_factor = config.rope_scaling.get("extrapolation_factor", 1)
            attn_factor = config.rope_scaling.get("attn_factor", 1)
            beta_fast = config.rope_scaling.get("beta_fast", 32)
            beta_slow = config.rope_scaling.get("beta_slow", 1)
            original_max_position_embeddings = config.original_max_position_embeddings
            assert (
                original_max_position_embeddings is not None
            ), "Using Yarn for rope scaling but original max_position_embeddings is None."

            def _yarn_find_correction_dim(
                num_rotations, dim, base, max_position_embeddings
            ):
                return (
                    dim
                    * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
                ) / (2 * math.log(base))

            def _yarn_find_correction_range(
                low_rot, high_rot, dim, base, max_position_embeddings
            ):
                low = math.floor(
                    _yarn_find_correction_dim(
                        low_rot, dim, base, max_position_embeddings
                    )
                )
                high = math.ceil(
                    _yarn_find_correction_dim(
                        high_rot, dim, base, max_position_embeddings
                    )
                )
                return max(low, 0), min(high, dim - 1)  # Clamp values just in case

            def _yarn_linear_ramp_mask(min_val, max_val, dim):
                if min_val == max_val:
                    max_val += 0.001  # Prevent singularity

                linear_func = (np.arange(dim, dtype=np.float32) - min_val) / (
                    max_val - min_val
                )
                return np.clip(linear_func, 0, 1)

            def _yarn_get_softmax_scale(config, scaling_factor):
                if scaling_factor <= 1 or config.model_type != "llama":
                    return 1.0
                return 0.1 * math.log(scaling_factor) + 1.0

            # yarn has 3 parts: interpolation, extrapolation and linear ramp
            inv_freq_extrapolation = 1.0 / (
                base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim)
            )  # (rot_dim/2)
            inv_freq_interpolation = inv_freq_extrapolation / scaling_factor

            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, rot_dim, base, original_max_position_embeddings
            )
            inv_freq_mask = (
                1 - _yarn_linear_ramp_mask(low, high, rot_dim // 2).astype(np.float32)
            ) * extrapolation_factor
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )

            t = np.arange(length, dtype=np.float32)  # (len)
            freqs = np.einsum("i,j->ij", t, inv_freq)  # (len, rot_dim/2)
            emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
            additional_softmax_scale = (
                _yarn_get_softmax_scale(config, scaling_factor) * attn_factor
            )
            master_cos = (
                np.cos(emb)[None, None, :, :] * additional_softmax_scale
            )  # (1,1,len,rot_dim)
            master_sin = (
                np.sin(emb)[None, None, :, :] * additional_softmax_scale
            )  # (1,1,len,rot_dim)

            rot_emb = np.concatenate((master_cos, master_sin), axis=1)

        elif config.rope_scaling["type"] == "linear":
            inv_freq = 1.0 / (
                base ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim)
            )  # (rot_dim/2)
            inv_freq /= config.rope_scaling["factor"]
            t = np.arange(length, dtype=np.float32)  # (len)
            freqs = np.einsum("i,j->ij", t, inv_freq)  # (len, rot_dim/2)
            emb = np.concatenate((freqs, freqs), axis=-1)  # (len, rot_dim)
            master_cos = np.cos(emb)[None, None, :, :]  # (1,1,len,rot_dim)
            master_sin = np.sin(emb)[None, None, :, :]  # (1,1,len,rot_dim)
            rot_emb = np.concatenate((master_cos, master_sin), axis=1)

        else:
            assert False, (
                f"Rope scaling only supports longrope, mrope, yarn and linear,"
                f'but got {config.rope_scaling["type"]}'
            )

    else:
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
        if isinstance(rot_emb, torch.Tensor):
            return rot_emb.to(dtype)
        return torch.from_numpy(rot_emb).to(dtype)
    if isinstance(rot_emb, torch.Tensor):
        return rot_emb.detach().cpu().numpy().astype(dtype)
    return rot_emb.astype(dtype)


def get_normalized_config(config_filepath):
    config_file = json.load(open(config_filepath, "r"))
    if config_file.get("llm", None) is None:
        if config_file["model_type"] == "llama":
            from models.llm_models.configuration_llama import (
                LlamaConfig as config_class,
            )
        elif config_file["model_type"] in ["qwen3", "qwen2"]:
            from models.llm_models.configuration_qwen import QwenConfig as config_class
        elif config_file["model_type"] in ["phi3", "phi4"]:
            from models.llm_models.configuration_phi import PhiConfig as config_class
        elif config_file["model_type"] in ["gemma1", "gemma2", "gemma3"]:
            from models.llm_models.configuration_gemma import (
                GemmaConfig as config_class,
            )
    else:
        if config_file["llm"]["model_type"] == "whisper_decoder":
            from models.llm_models.configuration_whisper import (
                WhisperConfig as config_class,
            )
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

    if config_file.get("llm", None) is None:
        if config_file["model_type"] == "llama":
            from models.llm_models.configuration_llama import (
                LlamaConfig as config_class,
            )
            from models.llm_models.modeling_llama import LlamaModelChunk as chunk_class
        elif config_file["model_type"] == "qwen2":
            from models.llm_models.configuration_qwen import QwenConfig as config_class
            from models.llm_models.modeling_qwen import Qwen2ModelChunk as chunk_class
        elif config_file["model_type"] == "qwen3":
            from models.llm_models.configuration_qwen import QwenConfig as config_class
            from models.llm_models.modeling_qwen import Qwen3ModelChunk as chunk_class
        elif config_file["model_type"] == "phi3":
            from models.llm_models.configuration_phi import PhiConfig as config_class
            from models.llm_models.modeling_phi import Phi3ModelChunk as chunk_class
        elif config_file["model_type"] == "phi4":
            from models.llm_models.configuration_phi import PhiConfig as config_class
            from models.llm_models.modeling_phi import Phi4ModelChunk as chunk_class
        elif config_file["model_type"] == "gemma2":
            from models.llm_models.configuration_gemma import (
                GemmaConfig as config_class,
            )
            from models.llm_models.modeling_gemma import Gemma2ModelChunk as chunk_class
        elif config_file["model_type"] == "gemma3":
            from models.llm_models.configuration_gemma import (
                GemmaConfig as config_class,
            )
            from models.llm_models.modeling_gemma import Gemma3ModelChunk as chunk_class
    else:
        if config_file["llm"]["model_type"] == "whisper_decoder":
            from models.llm_models.configuration_whisper import (
                WhisperConfig as config_class,
            )
            from models.llm_models.modeling_whisper import (
                WhisperDecoderModelChunk as decoder_class,
                WhisperEncoderModel as encoder_class,
            )

            chunk_class = [encoder_class, decoder_class]
    config = config_class(**config_file, response_handler=response_handler)
    if bypass_tokenizer:
        return config, weight_dir, chunk_class
    else:
        if config.tokenizer == "default":
            if config_file.get("llm", None) is None:
                if config_file["model_type"] in ["llama", "phi3"]:
                    from aot_utils.llm_utils.tokenizers_.tokenization_llama import (
                        LlamaTokenizer as tokenizer_class,
                    )
                elif config_file["model_type"] in ["qwen3", "qwen2"]:
                    from aot_utils.llm_utils.tokenizers_.tokenization_qwen2_fast import (
                        Qwen2TokenizerFast as tokenizer_class,
                    )
                elif config_file["model_type"] in ["gemma3", "gemma2", "gemma1"]:
                    from aot_utils.llm_utils.tokenizers_.tokenization_gemma_fast import (
                        GemmaTokenizerFast as tokenizer_class,
                    )
                elif config_file["model_type"] == "phi4":
                    from aot_utils.llm_utils.tokenizers_.tokenization_gpt2 import (
                        GPT2Tokenizer as tokenizer_class,
                    )
            else:
                if config_file["llm"]["model_type"] == "whisper_decoder":
                    from aot_utils.llm_utils.tokenizers_.tokenization_whisper import (
                        WhisperTokenizer as tokenizer_class,
                    )
        else:
            if config.tokenizer == "llama":
                from aot_utils.llm_utils.tokenizers_.tokenization_llama import (
                    LlamaTokenizer as tokenizer_class,
                )
            elif config.tokenizer == "qwen2":
                from aot_utils.llm_utils.tokenizers_.tokenization_qwen2 import (
                    Qwen2Tokenizer as tokenizer_class,
                )
            elif config.tokenizer == "qwen2_fast":
                from aot_utils.llm_utils.tokenizers_.tokenization_qwen2_fast import (
                    Qwen2TokenizerFast as tokenizer_class,
                )
            elif config.tokenizer == "gemma":
                from aot_utils.llm_utils.tokenizers_.tokenization_gemma import (
                    GemmaTokenizer as tokenizer_class,
                )
            elif config.tokenizer == "gemma_fast":
                from aot_utils.llm_utils.tokenizers_.tokenization_gemma_fast import (
                    GemmaTokenizerFast as tokenizer_class,
                )
            elif config.tokenizer == "gpt2":
                from aot_utils.llm_utils.tokenizers_.tokenization_gpt2 import (
                    GPT2Tokenizer as tokenizer_class,
                )
            elif config.tokenizer == "pretrained_fast":
                from aot_utils.llm_utils.tokenizers_.tokenization_utils_fast import (
                    PreTrainedTokenizerFast as tokenizer_class,
                )
            elif config.tokenizer == "whisper_decoder":
                from aot_utils.llm_utils.tokenizers_.tokenization_whisper import (
                    WhisperTokenizer as tokenizer_class,
                )

        return config, weight_dir, tokenizer_class, chunk_class
