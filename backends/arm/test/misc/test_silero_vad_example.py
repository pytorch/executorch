# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from backends.arm.test.tester.arm_tester import count_program_io_kinds
from examples.arm.silero_vad_example_ethos_u.model_export import (
    export_silero_vad_ethos_u as export_module,
)
from examples.arm.silero_vad_example_ethos_u.model_export.export_silero_vad_ethos_u import (
    calibrate_model,
    collect_inputs,
    get_state_qparams,
    quantize_model,
    StatefulSileroVAD,
)
from examples.arm.silero_vad_example_ethos_u.runtime.compare_vad_probs import compare
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from torch.export.graph_signature import InputKind, OutputKind


class ToySileroCore(torch.nn.Module):
    def forward(
        self, audio: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_state = state + audio.mean()
        probability = new_state.mean(dim=2)[:1]
        return probability, new_state


class CalibrationStateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(1))
        self.observed_states: list[torch.Tensor] = []

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        state = self.get_buffer("state")
        self.observed_states.append(state.clone())
        state.add_(1)
        return audio


def _contains_copy(graph_module: torch.fx.GraphModule) -> bool:
    return any(
        node.op == "call_function" and "copy_" in str(node.target)
        for node in graph_module.graph.nodes
    )


def _write_probabilities(path: Path, probabilities: Iterable[float]) -> None:
    values = tuple(probabilities)
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def test_stateful_silero_uses_registered_state_buffer() -> None:
    audio = torch.ones(1, 16)
    exported_program = torch.export.export(
        StatefulSileroVAD(ToySileroCore(), ((1.0, 0), (1.0, 0))).eval(),
        (audio,),
        strict=False,
    )

    count_program_io_kinds(
        exported_program,
        {
            InputKind.USER_INPUT: 1,
            InputKind.BUFFER: 1,
        },
        {
            OutputKind.USER_OUTPUT: 1,
        },
    )
    assert "state" in {
        spec.target
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.BUFFER
    }
    assert "state" in exported_program.state_dict
    assert exported_program.state_dict["state"].dtype == torch.int8
    assert _contains_copy(exported_program.graph_module)


def test_state_qparams_cover_calibration_stream() -> None:
    audio = torch.ones(1, 16)

    qparams = get_state_qparams(ToySileroCore(), [[audio, audio]])

    expected_scale = 2.0 / (
        export_module.STATE_QUANT_MAX - export_module.STATE_QUANT_MIN
    )
    for scale, zero_point in qparams:
        assert scale == pytest.approx(expected_scale)
        assert zero_point == export_module.STATE_QUANT_MIN


def test_silero_quantization_preserves_state_buffer() -> None:
    audio = torch.ones(1, 16)
    quantizer = EthosUQuantizer(
        EthosUCompileSpec(target="ethos-u85-256", memory_mode="Shared_Sram")
    )
    quantizer.set_global(get_symmetric_quantization_config())

    exported_program = quantize_model(ToySileroCore().eval(), quantizer, [[audio]])

    input_kinds = [spec.kind for spec in exported_program.graph_signature.input_specs]
    output_kinds = [spec.kind for spec in exported_program.graph_signature.output_specs]
    assert input_kinds.count(InputKind.USER_INPUT) == 1
    assert output_kinds.count(OutputKind.USER_OUTPUT) == 1
    assert "state" in {
        spec.target
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.BUFFER
    }
    assert "state" in exported_program.state_dict
    assert exported_program.state_dict["state"].dtype == torch.int8
    assert _contains_copy(exported_program.graph_module)


def test_collect_inputs_preserves_stream_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        export_module,
        "load_wav_16khz_mono",
        lambda _: torch.ones(export_module.WINDOW_SIZE),
    )

    streams = collect_inputs(["first.wav", "second.wav"], max_frames=2)

    assert [len(stream) for stream in streams] == [1, 1]


def test_calibration_resets_state_between_streams() -> None:
    model = CalibrationStateModel()
    audio = torch.ones(1, 16)

    calibrate_model(model, [[audio, audio], [audio]])

    torch.testing.assert_close(
        torch.cat(model.observed_states), torch.tensor([0.0, 1.0, 0.0])
    )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_probability_comparison_rejects_non_finite_binary(
    tmp_path, invalid_value
) -> None:
    expected_path = tmp_path / "expected.bin"
    actual_path = tmp_path / "actual.bin"
    _write_probabilities(expected_path, [0.1, 0.9])
    _write_probabilities(actual_path, [0.1, invalid_value])

    with pytest.raises(AssertionError, match="Non-finite probability"):
        compare(expected_path, actual_path, None, 0.5, 0.05, 0.01, 0)


def test_probability_comparison_rejects_non_finite_log(tmp_path) -> None:
    expected_path = tmp_path / "expected.bin"
    actual_log_path = tmp_path / "fvp.log"
    _write_probabilities(expected_path, [0.1, 0.9])
    actual_log_path.write_text("PROB 0.000 0.100000 silence\nPROB 0.032 nan silence\n")

    with pytest.raises(AssertionError, match="Non-finite probability"):
        compare(expected_path, None, actual_log_path, 0.5, 0.05, 0.01, 0)


def test_probability_comparison_allows_threshold_mismatch_limit(tmp_path) -> None:
    expected_path = tmp_path / "expected.bin"
    actual_path = tmp_path / "actual.bin"
    _write_probabilities(expected_path, [0.1, 0.8, 0.7, 0.6, 0.1])
    _write_probabilities(actual_path, [0.1, 0.8, 0.7, 0.4, 0.1])

    compare(expected_path, actual_path, None, 0.5, 0.21, 0.05, 1)
