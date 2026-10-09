# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monitor target diagnostics and optionally render ETDump profiles."""

import argparse
import socket
import warnings
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = DEPLOY_DIR.parent
ETDUMP_PREFIX = b"ETDP"


def checksum(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def read_line(connection: socket.socket) -> bytes:
    line = bytearray()
    while True:
        byte = connection.recv(1)
        if not byte:
            raise ConnectionError("Target closed the diagnostic connection")
        if byte == b"\n":
            return bytes(line).rstrip(b"\r")
        line.extend(byte)


def read_etdump(connection: socket.socket, header: bytes) -> bytes:
    if len(header) != 20:
        raise RuntimeError(f"Malformed ETDump header: {header!r}")
    size = int(header[4:12], 16)
    expected_checksum = int(header[12:20], 16)
    encoded = read_line(connection)
    if len(encoded) != size * 2:
        raise RuntimeError("Malformed ETDump payload length")
    data = bytes.fromhex(encoded.decode("ascii"))
    if checksum(data) != expected_checksum:
        raise RuntimeError("ETDump checksum mismatch")
    return data


def print_performance_report(
    etdump: bytes, etrecord_path: Path, etdump_path: Path
) -> None:
    import executorch.backends.cortex_m.ops.operators  # noqa: F401
    from executorch.devtools import Inspector
    from executorch.devtools.inspector import TimeScale

    etdump_path.parent.mkdir(parents=True, exist_ok=True)
    etdump_path.write_bytes(etdump)
    etrecord = str(etrecord_path) if etrecord_path.exists() else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output Buffer not found.*")
        inspector = Inspector(
            etdump_data=etdump,
            etrecord=etrecord,
            source_time_scale=TimeScale.CYCLES,
            target_time_scale=TimeScale.CYCLES,
        )
    print(f"ETDump: {etdump_path}")
    inspector.print_data_tabular()


def parse_tcp(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port:
        raise ValueError("--tcp must be HOST:PORT")
    return host, int(port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp", default="127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--etrecord",
        type=Path,
        default=EXAMPLE_DIR / "artifacts/person_detection.etrecord",
    )
    parser.add_argument(
        "--etdump-output",
        type=Path,
        default=DEPLOY_DIR / "fvp-results/profile.etdp",
    )
    args = parser.parse_args()

    with socket.create_connection(parse_tcp(args.tcp), args.timeout) as connection:
        connection.settimeout(None)
        while True:
            line = read_line(connection)
            if line.startswith(ETDUMP_PREFIX):
                data = read_etdump(connection, line)
                print_performance_report(data, args.etrecord, args.etdump_output)
            else:
                print(line.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
