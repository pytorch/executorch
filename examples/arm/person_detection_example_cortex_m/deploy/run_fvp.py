# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the FVP with a webcam, image, or video connected to VSI4."""

import argparse
import os
import shutil
import socket
import struct
import subprocess  # nosec B404
import threading
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEPLOY_DIR.parent / "artifacts"
PTE_ADDRESS = 0x72000000
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3


def _read_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ConnectionError("FVP closed the VSI connection")
        result.extend(chunk)
    return bytes(result)


def _fvp_environment(fvp: str, vsi_port: int) -> tuple[Path, dict[str, str]]:
    executable = shutil.which(fvp)
    if executable is None:
        candidate = Path(fvp).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Could not find FVP executable: {fvp}")
        executable_path = candidate.resolve()
    else:
        executable_path = Path(executable).resolve()

    installation = next(
        (
            parent
            for parent in executable_path.parents
            if (parent / "python/lib/python3.9").is_dir()
        ),
        None,
    )
    if installation is None:
        raise RuntimeError(
            f"Could not locate the bundled Python runtime for {executable_path}"
        )

    environment = os.environ.copy()
    environment["PERSON_DETECTION_VSI_PORT"] = str(vsi_port)
    environment["PYTHONHOME"] = str(installation / "python")
    library_paths = [
        str(installation / "python/lib"),
        str(installation / "fmtplib"),
    ]
    existing_library_path = environment.get("LD_LIBRARY_PATH")
    if existing_library_path:
        library_paths.append(existing_library_path)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return executable_path, environment


class FrameServer(threading.Thread):
    def __init__(self, port: int, camera: int, input_path: Path | None):
        super().__init__(name="vsi-frame-server", daemon=True)
        self.port = port
        self.camera = camera
        self.input_path = input_path
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.error: Exception | None = None
        self.listener: socket.socket | None = None
        self.connection: socket.socket | None = None

    def stop(self) -> None:
        self.stop_requested.set()
        if self.listener is not None:
            self.listener.close()
        if self.connection is not None:
            self.connection.close()

    def _serve(self) -> None:  # noqa: C901
        try:
            import cv2  # type: ignore[import-not-found, import-untyped]
        except ImportError as error:
            raise RuntimeError("VSI video input requires opencv-python") from error

        still_image = None
        if self.input_path is not None:
            still_image = cv2.imread(str(self.input_path))
        source = None
        if still_image is None:
            source = cv2.VideoCapture(
                str(self.input_path) if self.input_path is not None else self.camera
            )
            if not source.isOpened():
                source.release()
                description = self.input_path or f"webcam {self.camera}"
                raise RuntimeError(f"Could not open {description}")

        sent_still = False
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", self.port))
        self.listener.listen(1)
        self.listener.settimeout(0.5)
        self.ready.set()
        connection = None
        try:
            while not self.stop_requested.is_set():
                try:
                    connection, _ = self.listener.accept()
                    self.connection = connection
                    break
                except TimeoutError:
                    continue
            if connection is None:
                return
            with connection:
                while not self.stop_requested.is_set():
                    requested_size = struct.unpack("!I", _read_exact(connection, 4))[0]
                    if requested_size != FRAME_SIZE:
                        connection.sendall(b"\x02")
                        raise RuntimeError(
                            f"Target requested {requested_size} bytes; "
                            f"expected {FRAME_SIZE}"
                        )
                    if still_image is not None:
                        if sent_still:
                            connection.sendall(b"\x01")
                            continue
                        frame = still_image
                        sent_still = True
                        captured = True
                    else:
                        assert source is not None
                        captured, frame = source.read()
                    if not captured:
                        connection.sendall(b"\x01")
                        continue
                    frame = cv2.resize(
                        frame,
                        (FRAME_WIDTH, FRAME_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    connection.sendall(b"\x00" + frame.tobytes())
        finally:
            if source is not None:
                source.release()
            if self.listener is not None:
                self.listener.close()
            self.connection = None

    def run(self) -> None:
        try:
            self._serve()
        except Exception as error:
            if not self.stop_requested.is_set():
                self.error = error
            self.ready.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elf", type=Path, default=DEPLOY_DIR / "build-fvp/person_detection"
    )
    parser.add_argument(
        "--pte", type=Path, default=ARTIFACTS_DIR / "person_detection.pte"
    )
    parser.add_argument("--fvp", default="FVP_Corstone_SSE-300_Ethos-U55")
    parser.add_argument("--port", type=int, default=5000, help="diagnostic UART port")
    parser.add_argument("--vsi-port", type=int, default=6004)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam index")
    source.add_argument("--input", type=Path, help="image or video file")
    args = parser.parse_args()
    if not args.elf.is_file() or not args.pte.is_file():
        raise FileNotFoundError("Build deploy and export the model first.")
    if args.input is not None and not args.input.is_file():
        raise FileNotFoundError(args.input)

    fvp_executable, environment = _fvp_environment(args.fvp, args.vsi_port)

    frames = FrameServer(args.vsi_port, args.camera, args.input)
    frames.start()
    if not frames.ready.wait(timeout=10) or frames.error is not None:
        raise RuntimeError("Could not start VSI frame server") from frames.error

    command = [
        str(fvp_executable),
        "--quiet",
        f"--data={args.pte}@0x{PTE_ADDRESS:x}",
        "-C",
        "ethosu.extra_args=--fast",
        "-C",
        "mps3_board.visualisation.disable-visualisation=0",
        "-C",
        f"mps3_board.v_path={DEPLOY_DIR / 'vsi'}",
        "-C",
        "mps3_board.uart0.shutdown_on_eot=0",
        "-C",
        "mps3_board.telnetterminal0.mode=raw",
        "-C",
        "mps3_board.telnetterminal0.start_telnet=0",
        "-C",
        f"mps3_board.telnetterminal0.start_port={args.port}",
        "-C",
        "mps3_board.telnetterminal1.start_telnet=0",
        "-C",
        "mps3_board.telnetterminal2.start_telnet=0",
        "-C",
        "mps3_board.telnetterminal5.start_telnet=0",
        "--application",
        str(args.elf),
    ]
    source_description = args.input or f"webcam {args.camera}"
    print(f"FVP video source: {source_description}")
    print(f"FVP UART0 diagnostics: tcp://127.0.0.1:{args.port}")
    try:
        subprocess.run(command, check=True, env=environment)  # nosec B603
    finally:
        frames.stop()
        frames.join(timeout=2)
    if frames.error is not None:
        raise RuntimeError("VSI frame server failed") from frames.error


if __name__ == "__main__":
    main()
