# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 launches using the INT4 family's compilation and tile selection."""

import functools

from executorch.backends.cuda.autotune.cuda_graph_timing import time_with_cuda_graph_us
from executorch.backends.cuda.autotune.launch_params import _size_hint
from executorch.backends.cuda.triton.kernels import quantized_gemm_utils
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _async_compile,
    _parallel_compile_available,
    _precompile_untuned_configs,
    _queue_untuned_configs,
    _QUEUING_ONLY,
    _tuned_kernel,
)
from torch.library import wrap_triton

# NVFP4 split and tile timing: median of three cold CUDA-graph means.
nvfp4_launch_bench = functools.partial(
    time_with_cuda_graph_us, statistic="mean", replays=3
)
autotune_do_bench = functools.partial(
    quantized_gemm_utils.autotune_do_bench, statistic="mean", replays=3
)


def launch_nvfp4_kernel(kernel, grid, args, m_argument, **constexprs):
    if _QUEUING_ONLY.get():
        if (
            not _parallel_compile_available()
            or _async_compile.active_mode.get() is None
        ):
            raise RuntimeError("candidate prepass lost its active compile pool")
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
    # The packed FP16x2 (HALF2) GEMV is timed only with native FP4. Its exact
    # software decode stays launchable, but on A100 it was at most ~6% faster
    # than the FP32 GEMV per kernel while raising GEMV autotune time ~1.6x.
    eligible = [
        config
        for config in configs
        if values.get("NATIVE", False) or not config.kwargs.get("HALF2", False)
    ]
    kept = [
        config
        for config in eligible
        if config.num_stages
        <= (int(values["K"]) + config.kwargs["BK"] * split - 1)
        // (config.kwargs["BK"] * split)
    ]
    return kept or [min(eligible, key=lambda config: config.num_stages)]
