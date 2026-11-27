# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, numpy_helper
from torch import nn


def _checkpoint_loader():
    return importlib.import_module(
        "examples.models.supertonic.loaders.checkpoint_loader"
    )


def _write_model(path, initializers, nodes=(), inputs=(), outputs=()) -> None:
    graph = helper.make_graph(
        list(nodes),
        "test_graph",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
    )
    onnx.save(helper.make_model(graph), path)


def test_extract_initializers_preserves_names_values_and_dtypes(tmp_path) -> None:
    float_weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    integer_weight = np.asarray([2, 4], dtype=np.int64)
    model_path = tmp_path / "weights.onnx"
    _write_model(
        model_path,
        [
            numpy_helper.from_array(float_weight, name="float_weight"),
            numpy_helper.from_array(integer_weight, name="integer_weight"),
        ],
    )

    initializers = _checkpoint_loader().extract_initializers(model_path)

    assert set(initializers) == {"float_weight", "integer_weight"}
    torch.testing.assert_close(
        initializers["float_weight"], torch.from_numpy(float_weight)
    )
    torch.testing.assert_close(
        initializers["integer_weight"], torch.from_numpy(integer_weight)
    )


def test_extract_initializers_rejects_duplicate_names(tmp_path) -> None:
    model_path = tmp_path / "duplicate.onnx"
    _write_model(
        model_path,
        [
            numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weight"),
            numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="weight"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate initializer.*weight"):
        _checkpoint_loader().extract_initializers(model_path)


@pytest.mark.parametrize(
    ("operator", "trans_b", "expected"),
    [
        ("MatMul", None, torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])),
        ("Gemm", 0, torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])),
        ("Gemm", 1, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
        ("Conv", None, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
    ],
)
def test_transform_initializer_converts_onnx_operator_layouts(
    operator: str, trans_b: int | None, expected: torch.Tensor
) -> None:
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    kwargs = {} if trans_b is None else {"trans_b": trans_b}

    transformed = _checkpoint_loader().transform_initializer(source, operator, **kwargs)

    torch.testing.assert_close(transformed, expected)


class _TinyCheckpointModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.conv = nn.Conv1d(1, 2, 3, bias=False)


def _write_tiny_checkpoint(path) -> dict[str, torch.Tensor]:
    linear_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    linear_bias = torch.tensor([0.25, -0.5])
    conv_weight = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    nodes = [
        helper.make_node("MatMul", ["linear_input", "source.linear"], ["linear_mm"]),
        helper.make_node("Add", ["linear_mm", "source.linear_bias"], ["linear_output"]),
        helper.make_node("Conv", ["conv_input", "source.conv"], ["conv_output"]),
    ]
    _write_model(
        path,
        [
            numpy_helper.from_array(linear_weight.numpy(), name="source.linear"),
            numpy_helper.from_array(linear_bias.numpy(), name="source.linear_bias"),
            numpy_helper.from_array(conv_weight.numpy(), name="source.conv"),
        ],
        nodes=nodes,
    )
    return {
        "linear.weight": linear_weight.T,
        "linear.bias": linear_bias,
        "conv.weight": conv_weight,
    }


def test_load_onnx_initializers_uses_explicit_mapping_and_operator_layouts(
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    expected = _write_tiny_checkpoint(model_path)
    module = _TinyCheckpointModule()

    _checkpoint_loader().load_onnx_initializers(
        module,
        model_path,
        {
            "linear.weight": "source.linear",
            "linear.bias": "source.linear_bias",
            "conv.weight": "source.conv",
        },
    )

    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize("trans_b", [0, 1])
def test_load_onnx_initializers_extracts_gemm_trans_b_layout(
    tmp_path, trans_b: int
) -> None:
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = expected.T if trans_b == 0 else expected
    model_path = tmp_path / f"gemm-trans-b-{trans_b}.onnx"
    _write_model(
        model_path,
        [numpy_helper.from_array(source.numpy(), name="source.weight")],
        nodes=[
            helper.make_node(
                "Gemm",
                ["input", "source.weight"],
                ["output"],
                transB=trans_b,
            )
        ],
    )
    module = nn.Linear(3, 2, bias=False)

    _checkpoint_loader().load_onnx_initializers(
        module, model_path, {"weight": "source.weight"}
    )

    torch.testing.assert_close(module.weight, expected)


def test_load_onnx_initializers_rejects_ambiguous_graph_layouts(tmp_path) -> None:
    model_path = tmp_path / "ambiguous-layout.onnx"
    _write_model(
        model_path,
        [
            numpy_helper.from_array(
                np.arange(4, dtype=np.float32).reshape(2, 2),
                name="source.weight",
            )
        ],
        nodes=[
            helper.make_node(
                "Gemm",
                ["gemm_input", "source.weight"],
                ["gemm_output"],
                transB=1,
            ),
            helper.make_node(
                "MatMul",
                ["matmul_input", "source.weight"],
                ["matmul_output"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="ambiguous operator layouts.*source.weight"):
        _checkpoint_loader().load_onnx_initializers(
            nn.Linear(2, 2, bias=False),
            model_path,
            {"weight": "source.weight"},
        )


def test_load_onnx_initializers_rejects_missing_source_weight(tmp_path) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(ValueError, match="missing initializer.*does.not.exist"):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "does.not.exist",
                "linear.bias": "source.linear_bias",
                "conv.weight": "source.conv",
            },
        )


def test_load_onnx_initializers_rejects_duplicate_source_mapping(tmp_path) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(
        ValueError, match="duplicate initializer mapping.*source.linear"
    ):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.linear",
                "linear.bias": "source.linear",
                "conv.weight": "source.conv",
            },
        )


def test_load_onnx_initializers_rejects_shape_mismatch(tmp_path) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(ValueError, match=r"shape mismatch.*linear\.weight"):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.conv",
                "linear.bias": "source.linear_bias",
                "conv.weight": "source.linear",
            },
        )


def test_load_onnx_initializers_rejects_unmapped_model_weights(tmp_path) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(ValueError, match=r"unmapped model weights.*linear\.bias"):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.linear",
                "conv.weight": "source.conv",
            },
        )


def test_load_onnx_initializers_rejects_unknown_model_weights(tmp_path) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(ValueError, match=r"unknown model weights.*unused\.weight"):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.linear",
                "linear.bias": "source.linear_bias",
                "conv.weight": "source.conv",
                "unused.weight": "source.conv",
            },
        )


def test_load_onnx_initializers_rejects_unused_initializer_when_requested(
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    expected = _write_tiny_checkpoint(model_path)
    model = onnx.load(model_path)
    model.graph.initializer.append(
        numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="unused")
    )
    onnx.save(model, model_path)

    with pytest.raises(ValueError, match="unused initializer.*unused"):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.linear",
                "linear.bias": "source.linear_bias",
                "conv.weight": "source.conv",
            },
            reject_unused=True,
        )

    assert set(expected) == {"linear.weight", "linear.bias", "conv.weight"}


def test_load_onnx_initializers_rejects_unknown_allowed_unused_name(
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.onnx"
    _write_tiny_checkpoint(model_path)

    with pytest.raises(
        ValueError,
        match="allowed unused initializer not found.*misspelled",
    ):
        _checkpoint_loader().load_onnx_initializers(
            _TinyCheckpointModule(),
            model_path,
            {
                "linear.weight": "source.linear",
                "linear.bias": "source.linear_bias",
                "conv.weight": "source.conv",
            },
            reject_unused=True,
            allowed_unused={"misspelled"},
        )


_REAL_MODEL_DIR = os.environ.get("SUPERTONIC_MODEL_DIR")


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for published ONNX contract checks",
)
@pytest.mark.parametrize(
    ("filename", "inputs", "outputs"),
    [
        (
            "duration_predictor.onnx",
            {
                "text_ids": ("INT64", ("batch_size", "text_length")),
                "style_dp": ("FLOAT", ("batch_size", 8, 16)),
                "text_mask": ("FLOAT", ("batch_size", 1, "text_length")),
            },
            {"duration": ("FLOAT", ("Squeezeduration_dim_0",))},
        ),
        (
            "text_encoder.onnx",
            {
                "text_ids": ("INT64", ("batch_size", "text_length")),
                "style_ttl": ("FLOAT", ("batch_size", 50, 256)),
                "text_mask": ("FLOAT", ("batch_size", 1, "text_length")),
            },
            {
                "text_emb": (
                    "FLOAT",
                    ("Multext_emb_dim_0", 256, "Multext_emb_dim_2"),
                )
            },
        ),
        (
            "vector_estimator.onnx",
            {
                "noisy_latent": ("FLOAT", ("batch_size", 144, "latent_length")),
                "text_emb": ("FLOAT", ("batch_size", 256, "text_length")),
                "style_ttl": ("FLOAT", ("batch_size", 50, 256)),
                "latent_mask": ("FLOAT", ("batch_size", 1, "latent_length")),
                "text_mask": ("FLOAT", ("batch_size", 1, "text_length")),
                "current_step": ("FLOAT", ("batch_size",)),
                "total_step": ("FLOAT", ("batch_size",)),
            },
            {
                "denoised_latent": (
                    "FLOAT",
                    ("batch_size", 144, "latent_length"),
                )
            },
        ),
        (
            "vocoder.onnx",
            {"latent": ("FLOAT", ("batch_size", 144, "latent_length"))},
            {
                "wav_tts": (
                    "FLOAT",
                    ("Reshapewav_tts_dim_0", "Reshapewav_tts_dim_1"),
                )
            },
        ),
    ],
)
def test_published_onnx_graph_contracts(
    filename: str,
    inputs: dict[str, tuple[str, tuple[str | int, ...]]],
    outputs: dict[str, tuple[str, tuple[str | int, ...]]],
) -> None:
    model = onnx.load(
        Path(_REAL_MODEL_DIR) / "onnx" / filename, load_external_data=False
    )
    initializer_names = {initializer.name for initializer in model.graph.initializer}

    def contract(value) -> tuple[str, tuple[str | int, ...]]:
        tensor_type = value.type.tensor_type
        shape = tuple(
            (
                dimension.dim_param
                if dimension.HasField("dim_param")
                else dimension.dim_value
            )
            for dimension in tensor_type.shape.dim
        )
        return onnx.TensorProto.DataType.Name(tensor_type.elem_type), shape

    assert {
        value.name: contract(value)
        for value in model.graph.input
        if value.name not in initializer_names
    } == inputs
    assert {value.name: contract(value) for value in model.graph.output} == outputs


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for published stage checkpoint checks",
)
def test_stage_initializer_maps_cover_every_published_initializer() -> None:
    loader = _checkpoint_loader()
    config_path = Path(_REAL_MODEL_DIR) / "onnx" / "tts.json"
    from examples.models.supertonic.model.config import TTSConfig

    config = TTSConfig.from_json(config_path)
    cases = [
        (
            "duration_predictor.onnx",
            loader.DURATION_PREDICTOR_INITIALIZER_MAP,
            frozenset(),
            loader.load_duration_predictor,
            98,
        ),
        (
            "text_encoder.onnx",
            loader.TEXT_ENCODER_INITIALIZER_MAP,
            frozenset(),
            loader.load_text_encoder,
            146,
        ),
        (
            "vector_estimator.onnx",
            loader.VECTOR_ESTIMATOR_INITIALIZER_MAP,
            loader.VECTOR_ESTIMATOR_GENERATED_INITIALIZERS,
            loader.load_vector_estimator,
            352,
        ),
        (
            "vocoder.onnx",
            loader.VOCODER_INITIALIZER_MAP,
            loader.VOCODER_GENERATED_INITIALIZERS,
            loader.load_vocoder,
            103,
        ),
    ]

    for filename, mapping, generated, load_stage, expected_count in cases:
        model_path = Path(_REAL_MODEL_DIR) / "onnx" / filename
        initializer_names = {
            initializer.name for initializer in onnx.load(model_path).graph.initializer
        }

        assert len(mapping) == expected_count
        assert len(set(mapping.values())) == expected_count
        assert set(mapping.values()).isdisjoint(generated)
        assert set(mapping.values()) | set(generated) == initializer_names
        stage = load_stage(model_path, config)
        assert set(stage.state_dict()) == set(mapping)
