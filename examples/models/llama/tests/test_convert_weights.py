# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys

import pytest
import torch
from executorch.examples.models.llama import convert_weights
from safetensors.torch import save_file


def test_hf_attention_weights_use_meta_rope_order(tmp_path):
    config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1}
    (tmp_path / "config.json").write_text(json.dumps(config))
    query = torch.arange(64, dtype=torch.float32).view(8, 8)
    key = torch.arange(32, dtype=torch.float32).view(4, 8)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": query,
            "model.layers.0.self_attn.k_proj.weight": key,
        },
        tmp_path / "model.safetensors",
    )
    output = tmp_path / "converted.pth"
    convert_weights.convert_hf_to_meta(str(tmp_path), str(output))
    checkpoint = torch.load(output, weights_only=True)
    torch.testing.assert_close(
        checkpoint["layers.0.attention.wq.weight"], query[[0, 2, 1, 3, 4, 6, 5, 7]]
    )
    torch.testing.assert_close(
        checkpoint["layers.0.attention.wk.weight"], key[[0, 2, 1, 3]]
    )


def test_hf_tied_embeddings_remain_shared(tmp_path):
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    embeddings = torch.arange(80, dtype=torch.float32).view(10, 8)
    save_file({"model.embed_tokens.weight": embeddings}, tmp_path / "model.safetensors")
    output = tmp_path / "converted.pth"
    convert_weights.convert_hf_to_meta(str(tmp_path), str(output))
    checkpoint = torch.load(output, weights_only=True)
    torch.testing.assert_close(checkpoint["tok_embeddings.weight"], embeddings)
    assert (
        checkpoint["output.weight"].data_ptr()
        == checkpoint["tok_embeddings.weight"].data_ptr()
    )


def test_hf_conversion_can_emit_llama_params(tmp_path):
    config = {
        "hidden_size": 8,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "rope_scaling": {"factor": 32},
        "vocab_size": 10,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(
        {"model.embed_tokens.weight": torch.zeros(10, 8)},
        tmp_path / "model.safetensors",
    )
    params = tmp_path / "params.json"
    convert_weights.convert_hf_to_meta(
        str(tmp_path), str(tmp_path / "weights.pth"), params_output=str(params)
    )
    actual = json.loads(params.read_text())
    assert actual["dim"] == 8
    assert actual["hidden_dim"] == 32
    assert actual["n_heads"] == 2
    assert actual["n_kv_heads"] == 1
    assert actual["n_layers"] == 2
    assert actual["use_scaled_rope"]
    assert actual["rope_scale_factor"] == 32


def test_hf_to_meta_cli_emits_weights_and_params(tmp_path):
    config = {
        "hidden_size": 8,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "vocab_size": 10,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    embeddings = torch.arange(80, dtype=torch.float32).view(10, 8)
    save_file({"model.embed_tokens.weight": embeddings}, tmp_path / "model.safetensors")
    output = tmp_path / "weights.pth"
    params = tmp_path / "params.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "executorch.examples.models.llama.convert_weights",
            "hf-to-meta",
            str(tmp_path),
            str(output),
            "--params-output",
            str(params),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(output, weights_only=True)
    torch.testing.assert_close(checkpoint["tok_embeddings.weight"], embeddings)
    assert json.loads(params.read_text())["dim"] == 8


@pytest.mark.parametrize(
    "command", ["unsloth-to-meta", "load-and-convert-unsloth-to-meta"]
)
def test_unsloth_cli_exposes_state_dict_and_safetensors_conversion(tmp_path, command):
    weights = torch.arange(8, dtype=torch.float32).view(2, 4)
    state = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": weights}
    source = tmp_path / (
        "adapter.pth" if command == "unsloth-to-meta" else "adapter.safetensors"
    )
    if command == "unsloth-to-meta":
        torch.save(state, source)
    else:
        save_file(state, source)
    output = tmp_path / "converted.pth"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "executorch.examples.models.llama.convert_weights",
            command,
            str(source),
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(output, weights_only=True)
    torch.testing.assert_close(
        checkpoint["layers.0.attention.wq.lora_a.weight"], weights
    )
