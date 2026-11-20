# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 launches using the INT4 family's compilation and tile selection."""

import torch
import triton.testing
from executorch.backends.cuda.autotune.cuda_graph_timing import (
    graph_timing_active,
    ParkedTensorsNotRestored,
    time_with_cuda_graph_us,
    time_with_cuda_graph_us_counted,
)
from executorch.backends.cuda.autotune.launch_params import _size_hint
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _precompile_untuned_configs,
    _queue_untuned_configs,
    _QUEUING_ONLY,
    _tuned_kernel,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    _warn_graph_timing_fallback,
    AUTOTUNE_REP_MS,
    AUTOTUNE_WARMUP_MS,
)
from torch.library import wrap_triton


def nvfp4_launch_bench(kernel_call):
    return time_with_cuda_graph_us(kernel_call, statistic="mean", replays=3)


def autotune_do_bench(kernel_call, quantiles):
    """NVFP4 tile timing: median of three cold CUDA-graph means."""
    if graph_timing_active() and torch.cuda.current_stream().cuda_stream != 0:
        try:
            ms = (
                time_with_cuda_graph_us_counted(
                    kernel_call, statistic="mean", replays=3
                )
                / 1000.0
            )
        except ParkedTensorsNotRestored:
            raise
        except RuntimeError as error:
            _warn_graph_timing_fallback(error)
        else:
            return [ms] * len(quantiles) if quantiles else ms
    ms = triton.testing.do_bench(
        kernel_call,
        warmup=AUTOTUNE_WARMUP_MS,
        rep=AUTOTUNE_REP_MS,
        return_mode="mean",
    )
    return [ms] * len(quantiles) if quantiles else ms


def launch_nvfp4_kernel(kernel, grid, args, m_argument, **constexprs):
    if _QUEUING_ONLY.get():
        _queue_untuned_configs(kernel, args, constexprs)
        return
    _precompile_untuned_configs(kernel, args, constexprs)
    hinted = list(args)
    hinted[m_argument] = _size_hint(hinted[m_argument])
    selected = _tuned_kernel(kernel, tuple(hinted), constexprs)
    wrap_triton(selected)[grid](*args, **constexprs)


def prune_nvfp4_stages(configs, named_args, **kwargs):
    values = {**named_args, **kwargs}
    split = values.get("SPLIT_K", values.get("SPLIT", 1))
    eligible = [
        config
        for config in configs
        if not config.kwargs.get("HALF2", False) or values.get("NATIVE", False)
    ]
    kept = [
        config
        for config in eligible
        if config.num_stages
        <= (int(values["K"]) + config.kwargs["BK"] * split - 1)
        // (config.kwargs["BK"] * split)
    ]
    return kept or [min(eligible, key=lambda config: config.num_stages)]
