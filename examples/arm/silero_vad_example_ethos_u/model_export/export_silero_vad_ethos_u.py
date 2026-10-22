# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import math
import os
import struct
import sys
import wave
from pathlib import Path
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from examples.models.silero_vad.export_silero_vad import (  # noqa: E402
    CONTEXT_SIZE,
    HIDDEN_DIM,
    INPUT_SIZE,
    load_model,
    SAMPLE_RATE,
    WINDOW_SIZE,
)
from executorch.backends.arm.ethosu import (  # noqa: E402
    EthosUCompileSpec,
    EthosUPartitioner,
)
from executorch.backends.arm.quantizer import (  # noqa: E402
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from executorch.exir.passes.init_mutable_pass import (  # noqa: E402
    InitializedMutableBufferPass,
)
from executorch.extension.export_util.utils import save_pte_program  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)

STATE_BUFFER_NAME = "state"
STATE_QUANT_MIN = -128
STATE_QUANT_MAX = 127
CalibrationStreams = list[list[torch.Tensor]]
StateQParams = tuple[tuple[float, int], tuple[float, int]]


def load_wav_16khz_mono(path: str) -> torch.Tensor:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    if num_frames == 0:
        raise ValueError(f"{path}: audio file is empty")
    if compression != "NONE":
        raise ValueError(f"{path}: compressed WAV files are not supported")
    if channels != 1:
        raise ValueError(f"{path}: expected mono audio, got {channels} channels")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got {8 * sample_width}-bit")

    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32)
    return samples / float(1 << 15)


def make_model_input(
    audio: torch.Tensor,
    frame_index: int,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = frame_index * WINDOW_SIZE
    chunk = audio[offset : offset + WINDOW_SIZE]
    model_input = torch.zeros(1, INPUT_SIZE, dtype=torch.float32)
    model_input[0, :CONTEXT_SIZE] = context
    model_input[0, CONTEXT_SIZE : CONTEXT_SIZE + chunk.numel()] = chunk

    new_context = context.clone()
    if chunk.numel() >= CONTEXT_SIZE:
        new_context = chunk[-CONTEXT_SIZE:].clone()
    elif chunk.numel() > 0:
        new_context = torch.cat([context[chunk.numel() :], chunk]).clone()

    return model_input, new_context


def count_frames(audio: torch.Tensor) -> int:
    return math.ceil(audio.numel() / WINDOW_SIZE)


class StatefulSileroVAD(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, state_qparams: StateQParams) -> None:
        super().__init__()
        self.model = model
        (self.hidden_scale, self.hidden_zero_point), (
            self.cell_scale,
            self.cell_zero_point,
        ) = state_qparams
        self.register_buffer(
            STATE_BUFFER_NAME,
            torch.stack(
                (
                    torch.full(
                        (1, HIDDEN_DIM),
                        self.hidden_zero_point,
                        dtype=torch.int8,
                    ),
                    torch.full((1, HIDDEN_DIM), self.cell_zero_point, dtype=torch.int8),
                )
            ),
            persistent=True,
        )

    def reset_state(self) -> None:
        state = cast(torch.Tensor, self.state)
        state[0].fill_(self.hidden_zero_point)
        state[1].fill_(self.cell_zero_point)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        hidden = (
            state[0].to(torch.float32) - self.hidden_zero_point
        ) * self.hidden_scale
        cell = (state[1].to(torch.float32) - self.cell_zero_point) * self.cell_scale
        probability, new_state = self.model(model_input, torch.stack((hidden, cell)))
        quantized_hidden = torch.clamp(
            torch.round(new_state[0] / self.hidden_scale + self.hidden_zero_point),
            STATE_QUANT_MIN,
            STATE_QUANT_MAX,
        ).to(torch.int8)
        quantized_cell = torch.clamp(
            torch.round(new_state[1] / self.cell_scale + self.cell_zero_point),
            STATE_QUANT_MIN,
            STATE_QUANT_MAX,
        ).to(torch.int8)
        state.copy_(torch.stack((quantized_hidden, quantized_cell)))
        return probability


def reset_state(
    model: torch.nn.Module, initial_state: torch.Tensor | None = None
) -> None:
    state = model.get_buffer(STATE_BUFFER_NAME)
    if not isinstance(state, torch.Tensor):
        raise RuntimeError("Expected stateful Silero model to contain a state buffer")
    if initial_state is None:
        state.zero_()
    else:
        state.copy_(initial_state)


def collect_inputs(
    audio_paths: list[str],
    max_frames: int,
) -> CalibrationStreams:
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    streams: CalibrationStreams = []
    num_frames = 0
    for audio_path in audio_paths:
        audio = load_wav_16khz_mono(audio_path)
        context = torch.zeros(CONTEXT_SIZE, dtype=torch.float32)
        stream: list[torch.Tensor] = []

        for frame_index in range(count_frames(audio)):
            model_input, context = make_model_input(audio, frame_index, context)
            stream.append(model_input)
            num_frames += 1
            if num_frames >= max_frames:
                break

        if stream:
            streams.append(stream)
        if num_frames >= max_frames:
            break

    return streams


def get_state_qparams(
    model: torch.nn.Module, calibration_streams: CalibrationStreams
) -> StateQParams:
    min_values = torch.zeros(2, dtype=torch.float32)
    max_values = torch.zeros(2, dtype=torch.float32)
    with torch.no_grad():
        for stream in calibration_streams:
            state = torch.zeros(2, 1, HIDDEN_DIM, dtype=torch.float32)
            for model_input in stream:
                _, state = model(model_input, state)
                state_rows = state.reshape(2, -1)
                min_values = torch.minimum(min_values, state_rows.amin(dim=1))
                max_values = torch.maximum(max_values, state_rows.amax(dim=1))

    scales = torch.clamp(
        (max_values - min_values) / (STATE_QUANT_MAX - STATE_QUANT_MIN),
        min=torch.finfo(torch.float32).eps,
    )
    zero_points = torch.clamp(
        torch.round(STATE_QUANT_MIN - min_values / scales),
        STATE_QUANT_MIN,
        STATE_QUANT_MAX,
    )
    return (
        (float(scales[0]), int(zero_points[0])),
        (float(scales[1]), int(zero_points[1])),
    )


def calibrate_model(
    model: torch.nn.Module, calibration_streams: CalibrationStreams
) -> None:
    initial_state = model.get_buffer(STATE_BUFFER_NAME).detach().clone()
    try:
        with torch.no_grad():
            for stream in calibration_streams:
                reset_state(model, initial_state)
                for model_input in stream:
                    model(model_input)
    finally:
        reset_state(model, initial_state)


def run_stream(
    model: torch.nn.Module,
    audio_path: str,
    max_frames: int | None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    audio = load_wav_16khz_mono(audio_path)
    context = torch.zeros(CONTEXT_SIZE, dtype=torch.float32)
    probs: list[float] = []
    total_frames = count_frames(audio)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    reset_state(model, initial_state)
    with torch.no_grad():
        for frame_index in range(total_frames):
            model_input, context = make_model_input(audio, frame_index, context)
            prob = model(model_input)
            probs.append(float(prob.flatten()[0].item()))

    return torch.tensor(probs, dtype=torch.float32)


def write_float32_file(path: str, values: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = values.detach().cpu().flatten().tolist()
    with open(path, "wb") as output_file:
        output_file.write(struct.pack(f"<{len(data)}f", *data))


def ensure_quantized_out_variants() -> None:
    try:
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default.to_out_variant()
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default.to_out_variant()
        return
    except RuntimeError:
        pass

    configured_library = os.environ.get("EXECUTORCH_QUANTIZED_OPS_AOT_LIBRARY")
    if configured_library:
        library_path = Path(configured_library).expanduser().resolve()
        if not library_path.is_file():
            raise FileNotFoundError(f"Quantized ops library not found: {library_path}")
        torch.ops.load_library(str(library_path))
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default.to_out_variant()
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default.to_out_variant()
        return

    try:
        import executorch.extension.pybindings.portable_lib  # noqa: F401
    except ImportError:
        pass

    candidate_roots = [REPO_ROOT]
    executorch_pkg = importlib.import_module("executorch")
    for package_path in getattr(executorch_pkg, "__path__", []):
        resolved_package_path = Path(package_path).resolve()
        if resolved_package_path.parent.name == "src":
            candidate_roots.append(resolved_package_path.parent.parent)

    suffixes = (".so", ".dylib", ".dll")
    for root in dict.fromkeys(candidate_roots):
        quantized_dir = root / "kernels" / "quantized"
        for suffix in suffixes:
            candidate = quantized_dir / f"libquantized_ops_aot_lib{suffix}"
            if candidate.exists():
                torch.ops.load_library(str(candidate))
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default.to_out_variant()
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default.to_out_variant()
                return

    raise RuntimeError(
        "Could not load quantized q/dq out variants. Build or install the "
        "quantized AOT op library before exporting this example."
    )


def quantize_model(
    model: torch.nn.Module,
    quantizer: EthosUQuantizer,
    calibration_streams: CalibrationStreams,
):
    if not calibration_streams or not calibration_streams[0]:
        raise ValueError("No calibration examples were produced from the input audio")

    state_qparams = get_state_qparams(model, calibration_streams)
    print(
        "Recurrent state quantization: "
        f"hidden=({state_qparams[0][0]:.8f}, {state_qparams[0][1]}), "
        f"cell=({state_qparams[1][0]:.8f}, {state_qparams[1][1]})"
    )
    stateful_model = StatefulSileroVAD(model, state_qparams).eval()
    example_input = (calibration_streams[0][0],)
    exported_model = torch.export.export(
        stateful_model,
        example_input,
        strict=False,
    ).module()

    prepared_model = prepare_pt2e(exported_model, quantizer)

    print("\nCalibrating the model...")
    calibrate_model(prepared_model, calibration_streams)

    quantized_model = convert_pt2e(prepared_model)
    return torch.export.export(
        quantized_model,
        example_input,
        strict=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Silero VAD for Ethos-U")
    parser.add_argument(
        "--jit-model",
        type=str,
        required=True,
        help="Path to the Silero silero_vad.jit file.",
    )
    parser.add_argument(
        "--calibration-audio",
        nargs="+",
        required=True,
        help="One or more 16 kHz mono 16-bit PCM WAV files for PTQ calibration.",
    )
    parser.add_argument(
        "--validation-audio",
        type=str,
        default=None,
        help="16 kHz mono 16-bit PCM WAV used to generate expected_probs.bin.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./silero_vad_ethos_u.pte",
        help="Path to save the Ethos-U lowered ExecuTorch program.",
    )
    parser.add_argument(
        "--expected-output-path",
        type=str,
        default="./expected_probs.bin",
        help="Path to write reference float32 probabilities for validation.",
    )
    parser.add_argument(
        "--num-calibration-frames",
        type=int,
        default=256,
        help="Maximum number of audio frames to use for calibration.",
    )
    parser.add_argument(
        "--num-validation-frames",
        type=int,
        default=0,
        help="Maximum validation frames. Use 0 to process the whole WAV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading Silero VAD JIT model...")
    model = load_model(args.jit_model).eval()

    compile_spec = EthosUCompileSpec(
        target="ethos-u85-256",
        memory_mode="Shared_Sram",
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    calibration_streams = collect_inputs(
        args.calibration_audio,
        args.num_calibration_frames,
    )

    with torch.no_grad():
        quantized_program = quantize_model(model, quantizer, calibration_streams)

    validation_audio = args.validation_audio or args.calibration_audio[0]
    validation_frames = (
        None if args.num_validation_frames == 0 else args.num_validation_frames
    )
    expected_probs = run_stream(
        quantized_program.module(),
        validation_audio,
        validation_frames,
        quantized_program.state_dict[STATE_BUFFER_NAME].detach().clone(),
    )
    write_float32_file(args.expected_output_path, expected_probs)
    print(f"Wrote expected probabilities to {args.expected_output_path}")

    print("\nLowering to Ethos-U85...")
    partitioner = EthosUPartitioner(compile_spec)
    edge_program = to_edge_transform_and_lower(
        programs=quantized_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )
    ensure_quantized_out_variants()
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=False,
            passes=[InitializedMutableBufferPass([STATE_BUFFER_NAME])],
        )
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_pte_program(executorch_program, args.output_path)
    print(f"\nExported model saved to {args.output_path}")
