# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA streams for running delegates on a stream the caller chooses.

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

Limits for now: only streams on CUDA device 0 can be selected, and the CUDA
delegate refuses a method that uses CUDA graphs inside a block. This module
does not need PyTorch; ``torch.cuda.Stream`` works because it has
``__cuda_stream__``. The default stream (handle 0) cannot be selected.

As in PyTorch, the selection belongs to the thread, not to an asyncio task, so
do not await or yield inside a block. A stream must stay alive while a block
selects it, unless the block selects that very ``Stream`` object.
"""

import contextlib
import operator
import threading
import weakref
from typing import Any, Optional, Tuple


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
        """Destroys the stream. Refused while a block selects this Stream; a block
        that selects its handle through another wrapper is not counted."""
        self._native.close()

    def __reduce__(self) -> Any:
        # A copy would only alias this stream and close together with it.
        raise TypeError("A CUDA stream cannot be copied or pickled")


class ExternalStream:
    """A CUDA stream someone else created and keeps alive; never destroyed here.

    ``device`` is the CUDA device it belongs to, if known; stream() refuses a
    device other than 0.
    """

    def __init__(self, ptr: int, device: Optional[int] = None) -> None:
        self.device = device
        self.cuda_stream = ptr

    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, self.cuda_stream)


def _target(s: Any) -> Any:
    # An owned stream is checked and counted under its own lock on entry, so a
    # closed one is refused and an entered one cannot be closed.
    if isinstance(s, Stream):
        return s._native
    # The device is a torch.device (index), a cuda.core Device (device_id) or,
    # for ExternalStream, an int; CuPy keeps device_id on the stream, with -1
    # for unknown.
    device = getattr(s, "device", None)
    if device is None:
        device = getattr(s, "device_id", None)
    raw: Any = getattr(device, "index", getattr(device, "device_id", device))
    index: Optional[int] = None
    try:
        index = operator.index(raw)
    except TypeError:
        pass
    if index is not None and index > 0:
        raise RuntimeError("Only a stream on CUDA device 0 can be selected for now")
    version, handle = s.__cuda_stream__()
    if version != 0:
        raise ValueError(f"Unsupported __cuda_stream__ protocol version {version}")
    if not handle:
        # The CUDA delegate takes a selected stream as is, and refuses null.
        raise ValueError(
            "The default CUDA stream (handle 0) cannot be selected; "
            "select a stream created for this, such as cuda.Stream()"
        )
    return handle


class _Counter:
    # Kept apart from _StreamScope, so a finalizer that holds it does not
    # keep a dropped scope object, and its stream selection, alive.
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()

    def add(self, delta: int) -> None:
        with self.lock:
            self.value += delta


class _Entry:
    __slots__ = ("scope", "__weakref__")

    def __init__(self) -> None:
        self.scope: Any = None


class _StreamScope:
    def __init__(self, s: Any) -> None:
        if not hasattr(s, "__cuda_stream__"):
            raise TypeError(
                "stream() takes an object with __cuda_stream__, such as "
                "executorch.runtime.cuda.Stream or torch.cuda.Stream"
            )
        self._stream = s
        # One native scope per entering thread, in thread-local storage, so
        # threads that share this object cannot replace each other's, and a
        # thread that ends inside a block frees its slot (its selection already
        # ended natively). _active counts slots on all threads.
        self._local = threading.local()
        self._active = _Counter()

    def __enter__(self) -> Any:
        if getattr(self._local, "entry", None) is not None:
            raise RuntimeError("This CUDA stream scope is already active")
        # Reserved before any producer code runs, so a re-entrant
        # __cuda_stream__ cannot enter twice.
        entry = _Entry()
        self._local.entry = entry
        self._active.add(1)
        weakref.finalize(entry, self._active.add, -1)
        del entry
        scope = None
        try:
            scope = _bindings()._CallerStreamScope(_target(self._stream))
            scope.__enter__()
            self._local.entry.scope = scope
        except BaseException:
            self._local.entry = None
            if scope is not None:
                with contextlib.suppress(Exception):
                    scope.__exit__(None, None, None)
            raise
        return self._stream

    def __exit__(self, *exc_info: Any) -> None:
        entry = getattr(self._local, "entry", None)
        if entry is None or entry.scope is None:
            with self._active.lock:
                others = self._active.value
            if others:
                raise RuntimeError(
                    "A CUDA stream scope must end on the thread that started it"
                )
            return
        # Raises when blocks end out of order; the scope is kept, so a later
        # exit on this thread returns once the native side has ended it.
        entry.scope.__exit__(None, None, None)
        self._local.entry = None


def stream(s: Any) -> _StreamScope:
    """Makes ``s`` the CUDA stream for delegates run by this thread in the block.

    ``s`` is a Stream, an ExternalStream, or any object with ``__cuda_stream__``.
    The selection belongs to the thread, so blocks must end on the thread that
    started them, innermost first. An exit out of order is refused, and that
    scope then ends when the scopes started after it end. Until a block ends on
    its own thread, that thread keeps the stream selected, and a Stream it
    selects cannot be closed. Only CUDA device 0 is supported for now: a stream
    whose ``device`` is known to be another one is refused, and so is the
    default stream (handle 0). A method that uses CUDA graphs fails inside a
    block. Any other stream, including a wrapper of a Stream's handle, must stay
    alive while selected.
    """
    return _StreamScope(s)
