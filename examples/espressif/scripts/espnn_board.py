# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Board helpers for the ESP-NN examples: chip detection, ESP-IDF build and
flash of examples/espressif/espnn_runner, serial capture with reset, and
parsing of the runner's result lines. Needs an activated ESP-IDF shell.
"""

from __future__ import annotations

import fcntl
import os
import re
import select
import struct
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
RUNNER_DIR = EXAMPLES_DIR / "espnn_runner"
EXECUTORCH_ROOT = EXAMPLES_DIR.parents[1]
TIOCM_DTR = 0x002
TIOCM_RTS = 0x004
FIELDS = re.compile(r"(\w+)=(\S+)")
APP_PARTITION_BYTES = {4: 0x3F0000, 8: 0x7F0000, 16: 15 * 1024 * 1024}


def idf_command() -> list[str]:
    idf_path = os.environ.get("IDF_PATH")
    if not idf_path:
        raise RuntimeError("IDF_PATH is not set; run . $IDF_PATH/export.sh first")
    python = os.environ.get("IDF_PYTHON_ENV_PATH")
    interpreter = f"{python}/bin/python" if python else "python3"
    return [interpreter, f"{idf_path}/tools/idf.py"]


def tool_env() -> dict[str, str]:
    """ESP-IDF tools run in their own venv; a PYTHONPATH pointing at the torch
    environment would shadow its pinned packages (click, esptool, ...)."""
    return {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")
    }


def esptool_command() -> list[str]:
    python = os.environ.get("IDF_PYTHON_ENV_PATH")
    return [f"{python}/bin/python" if python else "python3", "-m", "esptool"]


@dataclass
class ChipInfo:
    target: str
    description: str
    flash_mb: int

    @property
    def app_partition_bytes(self) -> int:
        return APP_PARTITION_BYTES.get(self.flash_mb, APP_PARTITION_BYTES[4])


def detect_chip(port: str) -> ChipInfo:
    output = subprocess.run(
        esptool_command() + ["--port", port, "flash_id"],
        capture_output=True,
        text=True,
        timeout=60,
        env=tool_env(),
    ).stdout
    match = re.search(r"Chip (?:is|type:)\s*(ESP32[-\w]*)([^\n]*)", output)
    if not match:
        raise RuntimeError(f"Could not identify the chip on {port}:\n{output[-800:]}")
    name = match.group(1).upper()
    family = re.match(r"ESP32-(S2|S3|C2|C3|C5|C6|C61|H2|H4|P4)\b", name)
    target = f"esp32{family.group(1).lower()}" if family else "esp32"
    flash = re.search(r"Detected flash size:\s*(\d+)MB", output)
    return ChipInfo(
        target=target,
        description=(match.group(1) + match.group(2)).strip(),
        flash_mb=int(flash.group(1)) if flash else 4,
    )


def set_sdkconfig(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    output = []
    for line in lines:
        key = line.split("=", 1)[0].replace("# ", "").split(" is not set")[0].strip()
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(output) + "\n")


@dataclass
class RunnerBuild:
    target: str
    executorch: bool
    build_dir: Path
    executorch_build_dir: Path | None = None
    executorch_source_dir: Path | None = None
    extra_defaults: list[Path] = field(default_factory=list)

    def cmake_args(self) -> list[str]:
        defaults = [str(RUNNER_DIR / "sdkconfig.defaults")]
        if self.executorch:
            defaults.append(str(RUNNER_DIR / "sdkconfig.executorch"))
        defaults += [str(p) for p in self.extra_defaults]
        args = [
            f"-DIDF_TARGET={self.target}",
            f"-DSDKCONFIG={self.build_dir}/sdkconfig",
            f"-DSDKCONFIG_DEFAULTS={';'.join(defaults)}",
        ]
        if self.executorch and self.executorch_build_dir:
            args.append(f"-DESPNN_EXECUTORCH_BUILD_DIR={self.executorch_build_dir}")
        if self.executorch and self.executorch_source_dir:
            args.append(f"-DESPNN_EXECUTORCH_SOURCE_DIR={self.executorch_source_dir}")
        return args

    def configure(self, bundle: Path, iterations: int) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "CONFIG_ESPNN_RUNNER_BUNDLE": f'"{bundle.resolve()}"',
            "CONFIG_ESPNN_RUNNER_BENCH_ITERATIONS": str(iterations),
        }
        sdkconfig = self.build_dir / "sdkconfig"
        if sdkconfig.exists():
            set_sdkconfig(sdkconfig, values)
        else:
            overlay = self.build_dir / "sdkconfig.bundle"
            overlay.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
            if overlay not in self.extra_defaults:
                self.extra_defaults.append(overlay)

    def run_idf(
        self, *actions: str, port: str | None = None, log: Path | None = None
    ) -> str:
        command = (
            idf_command()
            + ["-C", str(RUNNER_DIR), "-B", str(self.build_dir)]
            + self.cmake_args()
        )
        if port:
            command += ["-p", port, "-b", "921600"]
        command += list(actions)
        result = subprocess.run(command, capture_output=True, text=True, env=tool_env())
        output = result.stdout + result.stderr
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(output)
        if result.returncode != 0:
            errors = [
                line
                for line in output.splitlines()
                if "error" in line.lower() or "FAILED" in line
            ]
            errors = errors or output.strip().splitlines()[-6:]
            raise RuntimeError(
                f"idf.py {' '.join(actions)} failed:\n" + "\n".join(errors[-15:])
            )
        return output

    def binary_size(self) -> int:
        image = self.build_dir / "espnn_runner.bin"
        return image.stat().st_size if image.exists() else 0


class SerialPort:
    def __init__(self, path: str, baud: int = 115200):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        attributes = termios.tcgetattr(self.fd)
        speed = getattr(termios, f"B{baud}")
        attributes[4] = attributes[5] = speed
        termios.tcsetattr(self.fd, termios.TCSANOW, attributes)

    def _lines(self, dtr: bool, rts: bool) -> None:
        bits = struct.unpack(
            "I", fcntl.ioctl(self.fd, termios.TIOCMGET, struct.pack("I", 0))
        )[0]
        bits = (bits | TIOCM_DTR) if dtr else (bits & ~TIOCM_DTR)
        bits = (bits | TIOCM_RTS) if rts else (bits & ~TIOCM_RTS)
        fcntl.ioctl(self.fd, termios.TIOCMSET, struct.pack("I", bits))

    def reset(self) -> None:
        self._lines(dtr=False, rts=True)
        time.sleep(0.1)
        self._lines(dtr=False, rts=False)

    def read(self, timeout: float) -> bytes:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return b""
        try:
            return os.read(self.fd, 65536)
        except OSError:
            return b""

    def close(self) -> None:
        os.close(self.fd)


@dataclass
class RunResult:
    chip: dict = field(default_factory=dict)
    cases: list[dict] = field(default_factory=list)
    passed: bool = False
    finished: bool = False
    log: str = ""
    last_error: str = ""


def parse_line(text: str, result: RunResult, on_case=None) -> None:
    if text.startswith("CHIP "):
        result.chip = dict(FIELDS.findall(text))
    elif text.startswith("ERROR:"):
        result.last_error = text[len("ERROR:") :].strip()
    elif text.startswith(("SUITE CASE", "ET CASE")):
        case = dict(FIELDS.findall(text))
        case["kind"] = "executorch" if text.startswith("ET CASE") else "standalone"
        if case.get("status") != "PASS" and result.last_error:
            case["error"] = result.last_error
        result.last_error = ""
        result.cases.append(case)
        if on_case:
            on_case(case)
    elif text.startswith("ESPNN RUNNER:"):
        result.finished = True
        result.passed = "PASS" in text


def capture(port: str, timeout: float, on_line=None, on_case=None) -> RunResult:
    result = RunResult()
    serial = SerialPort(port)
    serial.reset()
    deadline = time.time() + timeout
    pending = b""
    raw = bytearray()
    try:
        while time.time() < deadline and not result.finished:
            chunk = serial.read(0.5)
            if not chunk:
                continue
            raw += chunk
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").rstrip("\r")
                parse_line(text, result, on_case)
                if on_line:
                    on_line(text)
    finally:
        serial.close()
    result.log = raw.decode("utf-8", errors="replace")
    return result


def run_bundle(
    build: RunnerBuild,
    bundle: Path,
    port: str,
    iterations: int,
    timeout: float,
    log_dir: Path,
    on_line=None,
    on_case=None,
    on_status=None,
) -> RunResult:
    build.configure(bundle, iterations)
    if on_status:
        on_status("building")
    build.run_idf("build", log=log_dir / "build.log")
    if on_status:
        on_status("flashing")
    build.run_idf("flash", port=port, log=log_dir / "flash.log")
    if on_status:
        on_status("running on board")
    result = capture(port, timeout, on_line, on_case)
    (log_dir / "board.log").write_text(result.log)
    return result


if __name__ == "__main__":
    print(detect_chip(sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"))
