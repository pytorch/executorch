# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA streams for running delegates on a stream the caller chooses.

.. warning::

    This API is experimental and subject to change without notice.

Example::

    from executorch.runtime import cuda

    s = cuda.Stream()
    with cuda.stream(s):
        outputs = first.execute(inputs)  # e.g. a TensorRT delegate
        second.execute(outputs)  # e.g. a CUDA delegate in another program
    s.close()

Inside the block, the CUDA and TensorRT delegates queue their work on ``s``
instead of their own streams, and so do a program's copies between host and
GPU memory. A call then waits only for ``s``, not for the whole GPU: work that
other streams still have queued keeps running. A call with GPU inputs or
outputs, and ``ExecuTorchResult.to_host()``, still wait for the whole GPU, so
GPU data written on other streams stays safe.

Limits for now: a stream on a CUDA device other than 0 is refused, so is the
legacy default stream (handle 0, or 1 for ``cudaStreamLegacy``), and the CUDA
delegate refuses a method that uses CUDA graphs inside a block. The device is
read from the stream itself when ExecuTorch is built with, and runs on, CUDA
12.8 or newer; otherwise it goes by the device a wrapper reports. This module
does not need PyTorch; ``torch.cuda.Stream`` works because it has
``__cuda_stream__``.

As in PyTorch, the selection belongs to the thread, not to an asyncio task, so
do not await or yield inside a block. Blocks end on the thread that started
them, innermost first; a block left by a thread that ends, or by a dropped
scope object, ends on its own. A ``Stream`` cannot be closed while any block
selects its handle. Any other stream must stay alive while selected.
"""

import contextlib
import operator
import threading
import weakref
from typing import Any, Optional, Tuple, Union

# Every open Stream by handle, so a selection of its handle through any wrapper
# is checked and counted by the Stream itself.
_streams: "weakref.WeakValueDictionary[int, Stream]" = weakref.WeakValueDictionary()


def _bindings() -> Any:
    from executorch.extension.pybindings import _C as native

    if not hasattr(native, "_CallerStreamScope"):
        raise RuntimeError("CUDA streams need ExecuTorch built with CUDA support")
    return native


class Stream:
    """A non-blocking CUDA stream, destroyed by close() or garbage collection.

    ``device`` is the CUDA device to create it on; None means the current one.
    """

    def __init__(self, device: Optional[int] = None) -> None:
        self._native = _bindings()._CudaStream(-1 if device is None else device)
        self.device = self._native.device
        _streams[self._native.handle] = self

    @property
    def cuda_stream(self) -> int:
        """The stream handle, or 0 once closed."""
        return self._native.handle

    def __cuda_stream__(self) -> Tuple[int, int]:
        handle = self.cuda_stream
        if not handle:
            raise RuntimeError("This CUDA stream is closed")
        return (0, handle)

    def close(self) -> None:
        """Destroys the stream. Refused while a block selects its handle."""
        handle = self._native.handle
        self._native.close()
        if _streams.get(handle) is self:
            del _streams[handle]

    def __reduce__(self) -> Any:
        # A copy would only alias this stream and close together with it.
        raise TypeError("A CUDA stream cannot be copied or pickled")


class ExternalStream:
    """A CUDA stream someone else created and keeps alive; never destroyed here.

    ``device`` is the CUDA device it belongs to, if known, as an index or a
    string such as ``"cuda:1"``; stream() refuses a device other than 0.
    """

    def __init__(
        self, stream_ptr: int, device: Optional[Union[int, str]] = None
    ) -> None:
        self.device = device
        self.cuda_stream = stream_ptr

    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, self.cuda_stream)


def _device_index(s: Any) -> Optional[int]:
    # The device is a torch.device (index), a cuda.core Device (device_id), or
    # for ExternalStream an int or a string such as "cuda:1"; CuPy keeps
    # device_id on the stream, with -1 for unknown.
    device = getattr(s, "device", None)
    if device is None:
        device = getattr(s, "device_id", None)
    if isinstance(device, str):
        device = device.partition(":")[2] or None
        return None if device is None else int(device)
    raw: Any = getattr(device, "index", getattr(device, "device_id", device))
    try:
        return operator.index(raw)
    except TypeError:
        return None


def _target(s: Any) -> Any:
    # An owned stream is checked and counted under its own lock on entry, so a
    # closed one is refused and an entered one cannot be closed.
    if isinstance(s, Stream):
        return s._native
    version, handle = s.__cuda_stream__()
    if version != 0:
        raise ValueError(f"Unsupported __cuda_stream__ protocol version {version}")
    owner = _streams.get(handle)
    if owner is not None:
        return owner._native
    index = _device_index(s)
    if index is not None and index > 0:
        raise RuntimeError("Only a stream on CUDA device 0 can be selected for now")
    # 0 and cudaStreamLegacy (1) both name the legacy default stream, which
    # waits for every blocking stream, so selecting it would defeat the point
    # of choosing a stream. The CUDA delegate also refuses a null stream.
    if handle in (0, 1):
        raise ValueError(
            "The legacy default CUDA stream (handle 0 or 1) cannot be selected; "
            "select a stream created for this, such as cuda.Stream()"
        )
    return handle


_RESERVED = object()


class _StreamScope:
    def __init__(self, s: Any) -> None:
        if not hasattr(s, "__cuda_stream__"):
            raise TypeError(
                "stream() takes an object with __cuda_stream__, such as "
                "executorch.runtime.cuda.Stream or torch.cuda.Stream"
            )
        self._stream = s
        # One native scope per entering thread, so threads that share this
        # object cannot replace each other's. The native side keeps the
        # ordering and thread rules, and ends a block whose thread ended.
        self._local = threading.local()

    def __enter__(self) -> Any:
        if getattr(self._local, "scope", None) is not None:
            raise RuntimeError("This CUDA stream scope is already active")
        # Reserved before any producer code runs, so a re-entrant
        # __cuda_stream__ cannot enter twice.
        self._local.scope = _RESERVED
        scope = None
        try:
            scope = _bindings()._CallerStreamScope(_target(self._stream))
            scope.__enter__()
            self._local.scope = scope
        except BaseException:
            self._local.scope = None
            if scope is not None:
                with contextlib.suppress(Exception):
                    scope.__exit__(None, None, None)
            raise
        return self._stream

    def __exit__(self, *exc_info: Any) -> None:
        scope = getattr(self._local, "scope", None)
        if scope is None or scope is _RESERVED:
            raise RuntimeError("This CUDA stream scope is not active on this thread")
        try:
            # Refused when blocks end out of order; the native side then ends
            # this block when the blocks started after it end.
            scope.__exit__(None, None, None)
        finally:
            self._local.scope = None


def stream(s: Any) -> _StreamScope:
    """Makes ``s`` the CUDA stream for delegates run by this thread in the block.

    An exit out of order is refused, and that block then ends when the blocks
    started after it end. The module docstring lists the other rules.

    Args:
        s: A Stream, an ExternalStream, or any object with ``__cuda_stream__``.

    Returns:
        A context manager whose ``as`` value is ``s``.
    """
    return _StreamScope(s)
