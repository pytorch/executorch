"""Cold-L2 GPU timing with one AOTI invocation per captured graph.

ET captures a forward once and replays the resulting graph. Use one invocation
per capture for both timing controls. The supplied function must explicitly
use the captured CUDA stream. This timer alone does not choose a loader API
or establish that its completion handling is capture compatible.
"""
import contextlib
import math
import torch

_flush_buffers = {}
TIMING_POLICY = 'one AOTI call per capture; cold L2 before events; minimum of25 replay GPU event intervals'


def time_single_aoti_capture_us(fn):
    stream = torch.cuda.current_stream()
    if stream.cuda_stream == 0:
        raise RuntimeError('AOTI capture requires a non-default CUDA stream')
    device = torch.cuda.current_device()
    if device not in _flush_buffers:
        size = torch.cuda.get_device_properties(device).L2_cache_size
        _flush_buffers[device] = torch.empty(max(size, 1) // 4,
                                           dtype=torch.int32, device=device)
    flush = _flush_buffers[device]
    # Finish loader initialization and pending model completion outside capture.
    fn()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    retained_output = None
    try:
        with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
            flush.zero_()
            start.record()
            retained_output = fn()
            end.record()
        intervals = []
        for _ in range(25):
            graph.replay()
            end.synchronize()
            intervals.append(start.elapsed_time(end) * 1000.0)
        if not all(math.isfinite(v) and v > 0 for v in intervals):
            raise RuntimeError('Invalid AOTI replay GPU event interval')
        return min(intervals)
    finally:
        with contextlib.suppress(RuntimeError):
            graph.reset()
        del graph, retained_output
