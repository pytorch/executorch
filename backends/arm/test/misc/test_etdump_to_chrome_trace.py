# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import struct
import sys

import pytest

from executorch.backends.arm.scripts import etdump_to_chrome_trace
from executorch.devtools.etdump import schema_flatcc as flatcc
from executorch.devtools.etdump.serialize import serialize_to_etdump_flatcc


@pytest.fixture
def etdump_path(tmp_path):
    event = flatcc.ProfileEvent(
        name="op",
        chain_index=0,
        instruction_id=0,
        delegate_debug_id_int=-1,
        delegate_debug_id_str="",
        delegate_debug_metadata=b"",
        start_time=1000,
        end_time=1500,
    )
    dump = flatcc.ETDumpFlatCC(
        version=0,
        run_data=[
            flatcc.RunData(
                name="Execute",
                bundled_input_index=-1,
                allocators=[],
                events=[
                    flatcc.Event(
                        profile_event=event, allocation_event=None, debug_event=None
                    )
                ],
            )
        ],
    )
    path = tmp_path / "etdump.bin"
    data = serialize_to_etdump_flatcc(dump)
    path.write_bytes(struct.pack("<I", len(data)) + data)
    return path


@pytest.mark.parametrize(
    "scale,frequency,timestamp,duration",
    [
        ("cycles", "500000000", 2.0, 1.0),
        ("cycles", "1000000000", 1.0, 0.5),
        ("ns", None, 1.0, 0.5),
        ("us", None, 1000.0, 500.0),
        ("ms", None, 1000000.0, 500000.0),
        ("s", None, 1000000000.0, 500000000.0),
    ],
)
def test_trace_uses_microseconds(
    monkeypatch, tmp_path, etdump_path, scale, frequency, timestamp, duration
):
    output = tmp_path / "trace.json"
    command = [
        "etdump_to_chrome_trace",
        "--etdump_path",
        str(etdump_path),
        "--output",
        str(output),
        "--source_time_scale",
        scale,
    ]
    if frequency is not None:
        command += ["--clock_frequency_hz", frequency]
    monkeypatch.setattr(sys, "argv", command)

    etdump_to_chrome_trace.main()

    events = json.loads(output.read_text())["traceEvents"]
    event = next(event for event in events if event["ph"] == "X")
    assert event["ts"] == pytest.approx(timestamp)
    assert event["dur"] == pytest.approx(duration)


@pytest.mark.parametrize("frequency", [None, "0", "-1", "nan", "inf"])
def test_cycle_trace_requires_valid_frequency(monkeypatch, tmp_path, frequency, capsys):
    command = [
        "etdump_to_chrome_trace",
        "--etdump_path",
        str(tmp_path / "missing.bin"),
        "--output",
        str(tmp_path / "trace.json"),
        "--source_time_scale",
        "cycles",
    ]
    if frequency is not None:
        command += ["--clock_frequency_hz", frequency]
    monkeypatch.setattr(sys, "argv", command)

    with pytest.raises(SystemExit) as error:
        etdump_to_chrome_trace.main()

    assert error.value.code == 2
    assert "positive finite --clock_frequency_hz" in capsys.readouterr().err
