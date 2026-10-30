# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VSI4 video-input adapter for the person-detection FVP runner."""

from __future__ import annotations

import os
import socket
import struct


IRQ_Status = 0
Timer_Control = 0
Timer_Interval = 0
DMA_Control = 0
Regs = [0] * 64
Data = bytearray()
Server: socket.socket | None = None

STATUS_ACTIVE = 1 << 0
STATUS_BUFFER_EMPTY = 1 << 1
STATUS_UNDERFLOW = 1 << 4
STATUS_END_OF_STREAM = 1 << 5


def _read_exact(size):
    assert Server is not None
    data = bytearray()
    while len(data) < size:
        chunk = Server.recv(size - len(data))
        if not chunk:
            raise ConnectionError("VSI frame server closed the connection")
        data.extend(chunk)
    return data


def init():
    global Server
    port = int(os.environ.get("PERSON_DETECTION_VSI_PORT", "6004"))
    Server = socket.create_connection(("127.0.0.1", port), timeout=30)
    Server.settimeout(None)


def rdIRQ():
    return IRQ_Status


def wrIRQ(value):
    global IRQ_Status
    IRQ_Status = value
    return value


def wrTimer(index, value):
    global Timer_Control, Timer_Interval
    if index == 0:
        Timer_Control = value
    elif index == 1:
        Timer_Interval = value
    return value


def timerEvent():
    return


def wrDMA(index, value):
    global DMA_Control
    if index == 0:
        DMA_Control = value
    return value


def rdDataDMA(size):
    global Data
    assert Server is not None
    Server.sendall(struct.pack("!I", size))
    response = _read_exact(1)[0]
    if response == 0:
        Data = _read_exact(size)
        Regs[2] = 0
    elif response == 1:
        Data = bytearray(size)
        Regs[2] = STATUS_BUFFER_EMPTY | STATUS_END_OF_STREAM
    else:
        Data = bytearray(size)
        Regs[2] = STATUS_BUFFER_EMPTY | STATUS_UNDERFLOW
    return Data


def wrDataDMA(data, size):
    global Data
    Data = data[:size]


def rdRegs(index):
    return Regs[index]


def wrRegs(index, value):
    Regs[index] = value
    if index == 1:
        if value & 1:
            Regs[2] = STATUS_ACTIVE | STATUS_BUFFER_EMPTY
        elif not (Regs[2] & (STATUS_END_OF_STREAM | STATUS_UNDERFLOW)):
            Regs[2] = 0
    return value
