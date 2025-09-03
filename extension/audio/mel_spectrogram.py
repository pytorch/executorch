# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)

from torch.export import Dim, export, ExportedProgram


class WhisperAudioProcessor(nn.Module):
    r"""
    Computes Mel spectrograms from mono audio input.
    Same as HuggingFace WhisperFeatureExtractor, but implemented in PyTorch

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        stack_batch: bool = False,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        )
        self.stack_batch = stack_batch

    def get_mel_filters(
        self, sr: int, n_fft: int, n_mels: int = 128, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # Initialize the weights
        n_mels = int(n_mels)
        weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr, dtype=dtype)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = torch.linspace(min_mel, max_mel, n_mels + 2, dtype=dtype)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = (
            torch.log(torch.tensor(6.4, dtype=dtype)) / 27.0
        )  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = torch.diff(mel_f)
        ramps = torch.subtract(mel_f.unsqueeze(1), fftfreqs.unsqueeze(0))

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = torch.maximum(
                torch.tensor(0.0, dtype=dtype), torch.minimum(lower, upper)
            )

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])  # pyre-ignore[58]
        weights *= enorm[:, None]  # pyre-ignore[16]

        return weights

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`): Mono waveform input, tensor of (dynamic) shape [num_samples],

        Returns:
            torch.Tensor: Output of shape [1, feature_size, nb_max_frames * n_chunks]
            n_chunks is the number of chunks of `sampling_rate` samples in the input waveform.
            [1, 80, 3000] with default options and 1 chunk
        """
        n_chunks = (waveform.shape[0] - 1) // self.n_samples + 1
        waveform = F.pad(
            waveform,
            (0, self.n_samples * n_chunks - waveform.shape[0]),
            mode="constant",
            value=self.padding_value,
        )
        # torch._check(waveform.shape[0] % self.n_samples == 0)

        # Ideally we should do:
        # window = torch.hann_window(self.n_fft)
        # but this is not currently supported when lowering.
        # torch.hann_window has slightly better numerics (worst discrepancy is <1e-5 instead of 1e-4)
        window = 0.5 * (
            1
            - torch.cos(
                2
                * torch.pi
                * torch.linspace(0, self.n_fft - 1, self.n_fft, dtype=torch.float32)
                / self.n_fft
            )
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitudes = torch.abs(stft)[..., :-1] ** 2  # pyre-ignore[58]

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # torch._check(log_spec.numel() % (self.feature_size * self.nb_max_frames) == 0)
        log_spec = log_spec.reshape(self.feature_size, -1, self.nb_max_frames)
        log_spec = log_spec.transpose(0, 1)

        return log_spec


def export_processor():
    model = WhisperAudioProcessor()
    audio_tensor = torch.randn(480000)
    chunk_tensor = audio_tensor[:93680]
    with torch.no_grad():
        ep: ExportedProgram = torch.export.export(
            model, (chunk_tensor,), dynamic_shapes={"waveform": {0: Dim.AUTO}}, strict=True
        )
        logging.debug(ep)

        # to edge
        edge: EdgeProgramManager = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        logging.debug(edge.exported_program())

        # to executorch
        exec_prog = edge.to_executorch()
        output_file = "whisper_preprocess.pte"
        with open(output_file, "wb") as file:
            exec_prog.write_to_file(file)

        logging.debug("Done")


def main():
    export_processor()


if __name__ == "__main__":
    main()
