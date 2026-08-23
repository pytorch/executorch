# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform

import pytest
import torch

from examples.models.supertonic.export import common
from examples.models.supertonic.export import export_supertonic
from examples.models.supertonic.model.config import TTSConfig
from examples.models.supertonic.model.duration_predictor import DurationPredictor
from examples.models.supertonic.model.text_encoder import TextEncoder
from examples.models.supertonic.model.vector_estimator import VectorEstimator
from examples.models.supertonic.model.vocoder import Vocoder


def _has_mlx() -> bool:
    try:
        import executorch.backends.mlx.custom_ops  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not _has_mlx(),
    reason="Darwin with the ExecuTorch MLX backend is required",
)

BOUNDS = common.ExportBounds(text_max=4, latent_max=3)
VALID_LENGTHS = ((1, 1), (4, 1), (1, 3), (2, 2), (4, 3))


def _config() -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {"latent_dim": 2, "chunk_compress_factor": 2},
            "ae": {
                "sample_rate": 16000,
                "base_chunk_size": 4,
                "chunk_compress_factor": 1,
                "ldim": 2,
            },
            "dp": {"latent_dim": 2, "chunk_compress_factor": 2},
        }
    )


def _models(config: TTSConfig) -> dict[str, torch.nn.Module]:
    models = {
        "duration_predictor": DurationPredictor(
            config,
            vocab_size=256,
            channels=4,
            convnext_dilations=(1,),
            attention_layers=1,
            attention_heads=1,
            ff_channels=4,
            style_tokens=8,
            style_dim=16,
            hidden_dim=4,
        ),
        "text_encoder": TextEncoder(
            config,
            vocab_size=256,
            channels=256,
            convnext_dilations=(1,),
            attention_layers=1,
            attention_heads=1,
            ff_channels=4,
            style_tokens=50,
            style_attention_heads=2,
        ),
        "vector_estimator": VectorEstimator(
            config,
            hidden_channels=8,
            time_dim=4,
            time_hidden_channels=8,
            num_main_blocks=1,
            main_convnext_dilations=(1,),
            post_time_dilations=(1,),
            post_text_dilations=(1,),
            final_dilations=(1,),
            text_channels=256,
            style_tokens=50,
            style_channels=256,
            attention_heads=1,
            style_attention_heads=2,
            max_positions=1000,
        ),
        "vocoder": Vocoder(
            config,
            decoder_channels=8,
            decoder_dilations=(),
            decoder_expansion=2,
            head_hidden_channels=8,
        ),
    }
    return dict(common.convert_models_to_fp16(models))


def _inputs_at(
    config: TTSConfig, *, text_length: int, latent_length: int
) -> dict[str, tuple[torch.Tensor, ...]]:
    samples = common.example_inputs(config, BOUNDS)
    samples["duration_predictor"] = (
        samples["duration_predictor"][0][:, :text_length].contiguous(),
        samples["duration_predictor"][1],
        samples["duration_predictor"][2][:, :, :text_length].contiguous(),
    )
    samples["text_encoder"] = (
        samples["text_encoder"][0][:, :text_length].contiguous(),
        samples["text_encoder"][1],
        samples["text_encoder"][2][:, :, :text_length].contiguous(),
    )
    vector = samples["vector_estimator"]
    vector_inputs = (
        vector[0][:, :, :latent_length].contiguous(),
        vector[1][:, :, :text_length].contiguous(),
        vector[2],
        vector[3][:, :, :latent_length].contiguous(),
        vector[4][:, :, :text_length].contiguous(),
        vector[5],
        vector[6],
    )
    common.validate_vector_inputs(vector_inputs, config, BOUNDS)
    samples["vector_estimator"] = vector_inputs
    samples["vocoder"] = (samples["vocoder"][0][:, :, :latent_length].contiguous(),)
    return samples


@pytest.fixture(scope="module")
def exported():
    config = _config()
    programs = export_supertonic.export_programs(_models(config), config, BOUNDS)
    return config, programs


@pytest.fixture(scope="module")
def lowered(exported):
    config, programs = exported
    edge = export_supertonic.lower_to_mlx(
        programs,
        common.runtime_metadata(config, BOUNDS, text_vocabulary_size=256),
    )
    return config, edge


def test_all_four_methods_export_together_with_dynamic_fp16_contracts(
    exported,
) -> None:
    config, programs = exported

    assert set(programs) == set(common.METHOD_NAMES)
    for text_length, latent_length in VALID_LENGTHS:
        inputs = _inputs_at(
            config,
            text_length=text_length,
            latent_length=latent_length,
        )
        outputs = {
            name: programs[name].module()(*inputs[name]) for name in common.METHOD_NAMES
        }
        assert outputs["duration_predictor"].shape == (1,)
        assert outputs["text_encoder"].shape == (1, 256, text_length)
        assert outputs["vector_estimator"].shape == (1, 4, latent_length)
        assert outputs["vocoder"].shape == (1, latent_length * 8)
        assert all(output.dtype == torch.float16 for output in outputs.values())


def test_export_accepts_models_in_any_mapping_insertion_order() -> None:
    config = _config()
    models = _models(config)
    reversed_models = dict(reversed(tuple(models.items())))

    programs = export_supertonic.export_programs(reversed_models, config, BOUNDS)

    assert set(programs) == set(common.METHOD_NAMES)


def test_expected_tensor_ops_are_delegated_without_cpu_fallback(
    exported, lowered
) -> None:
    _, programs = exported
    _, edge = lowered
    expected_ops = {
        "duration_predictor": ("embedding", "linear"),
        "text_encoder": ("embedding", "matmul"),
        "vector_estimator": ("conv1d", "matmul", "softmax"),
        "vocoder": ("conv1d", "batch_norm", "where"),
    }

    assert edge.methods == set(common.METHOD_NAMES)
    for method_name in common.METHOD_NAMES:
        aten_targets = {
            str(node.target)
            for node in programs[method_name].graph.nodes
            if node.op == "call_function"
        }
        for expected in expected_ops[method_name]:
            assert any(expected in target for target in aten_targets), (
                method_name,
                expected,
                sorted(aten_targets),
            )

        edge_targets = [
            str(node.target)
            for node in edge.exported_program(method_name).graph.nodes
            if node.op == "call_function"
        ]
        assert sum("executorch_call_delegate" in target for target in edge_targets) == 1
        assert all(
            "executorch_call_delegate" in target
            or target == "<built-in function getitem>"
            for target in edge_targets
        )


def test_saved_multi_method_pte_reloads_and_runs_dynamic_lengths(
    lowered, tmp_path
) -> None:
    from executorch.runtime import Runtime, Verification

    config, edge = lowered
    et_program = export_supertonic.to_executorch(edge)
    assert not et_program._tensor_data
    pte_path = common.save_pte(et_program, tmp_path / "supertonic.pte")
    assert set(tmp_path.iterdir()) == {pte_path}
    program = Runtime.get().load_program(pte_path, verification=Verification.Minimal)

    metadata = common.runtime_metadata(
        config, BOUNDS, text_vocabulary_size=256
    )
    assert program.method_names == set(common.METHOD_NAMES) | set(metadata)
    for method_name, expected in metadata.items():
        actual = program.load_method(method_name).execute([])[0]
        assert actual == expected
        assert type(actual) is type(expected)

    from executorch.backends.mlx.pte_inspector import parse_executorch_program

    serialized = parse_executorch_program(pte_path.read_bytes())["program"]
    plans = {plan["name"]: plan for plan in serialized["execution_plan"]}
    assert set(plans) == program.method_names
    for method_name in common.METHOD_NAMES:
        assert len(plans[method_name].get("delegates", [])) == 1
    for method_name in metadata:
        assert plans[method_name].get("delegates", []) == []

    for text_length, latent_length in VALID_LENGTHS:
        inputs = _inputs_at(
            config,
            text_length=text_length,
            latent_length=latent_length,
        )
        outputs = {
            name: program.load_method(name).execute(list(inputs[name]))[0]
            for name in common.METHOD_NAMES
        }
        assert outputs["duration_predictor"].shape == (1,)
        assert outputs["text_encoder"].shape == (1, 256, text_length)
        assert outputs["vector_estimator"].shape == (1, 4, latent_length)
        assert outputs["vocoder"].shape == (1, latent_length * 8)
        assert all(output.dtype == torch.float16 for output in outputs.values())
