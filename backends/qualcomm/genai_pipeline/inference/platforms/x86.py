# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the x86 build of an inference runner on the host."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.export_utils import QnnConfig

logger = logging.getLogger(__name__)


class X86Platform:
    """Runs the x86 build of a runner on the host against the QNN SDK's x86 libraries.

    Its responsibility is the host process: ``setup`` locates the runner in
    ``qnn_config.build_folder`` and the QNN SDK's x86 libraries, and ``execute``
    launches it with them on ``LD_LIBRARY_PATH``, raising on a non-zero exit.
    Nothing is transferred: the runner reads host files by absolute path, so
    ``runner_path`` resolves them, ``runner_input_list`` rewrites an input
    list's entries to absolute paths, and ``add_file`` does nothing. It writes
    its outputs into the session's ``host_dir``, from which ``pull`` moves them.

    The emulator supports no shared buffers and decodes slowly, so it passes no
    target flags and limits generation, as the legacy flow does.

    Args:
        qnn_config: Settings supplying the x86 build folder.
    """

    # The emulator supports no shared buffers; x86 PTEs are compiled without them.
    target_flags: Tuple[str, ...] = ()
    # The emulator is slow, so generation is kept short as the legacy flow does.
    limits_generation: bool = True

    def __init__(self, qnn_config: "QnnConfig") -> None:
        self._qnn_config = qnn_config
        self._build_folder: Optional[Path] = None
        self._qnn_x86_lib: Optional[Path] = None
        self._runner: Optional[Path] = None
        self._host_dir: Optional[Path] = None

    def setup(self, runner: str, files: Sequence[Path], host_dir: Path) -> None:
        """Locate the runner in the build folder; nothing is transferred.

        Args:
            runner: The runner binary, relative to the build folder.
            files: Unused; the runner reads host files in place.
            host_dir: Directory the runner writes its outputs to.

        Raises:
            RuntimeError: If ``QNN_SDK_ROOT`` is not set.
        """
        qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
        if not qnn_sdk_root:
            raise RuntimeError(
                "QNN_SDK_ROOT is not set; the x86 runner loads the QNN SDK's x86 "
                "libraries"
            )
        self._qnn_x86_lib = Path(qnn_sdk_root) / "lib" / "x86_64-linux-clang"
        self._build_folder = Path(self._qnn_config.build_folder).resolve()
        self._runner = self._build_folder / runner
        self._host_dir = host_dir

    def add_file(self, path: Path) -> None:
        """Nothing to do: the runner reads ``path`` in place."""

    def runner_path(self, host_path: Path) -> str:
        """Return the file's absolute host path."""
        return str(host_path.resolve())

    def runner_input_list(self, host_path: Path) -> str:
        """Rewrite the list's entries to absolute paths and return the list's.

        The runner does not run from the list's directory, so each entry is
        prefixed with it, as the legacy flow does. Entries already absolute are
        kept, so rewriting the same list again changes nothing.
        """
        list_dir = host_path.resolve().parent

        def resolve(entry: str) -> str:
            if not entry or Path(entry).is_absolute():
                return entry
            return str(list_dir / entry)

        lines = host_path.read_text().splitlines()
        host_path.write_text(
            "".join(" ".join(map(resolve, line.split(" "))) + "\n" for line in lines)
        )
        return str(host_path.resolve())

    def output_dir(self) -> str:
        """Return the session's ``host_dir``."""
        return str(self._require_host_dir())

    def execute(self, args: Sequence[str]) -> None:
        """Run the runner on the host.

        Raises:
            RuntimeError: If the runner exits with a non-zero code.
        """
        self._require_host_dir()
        command = " ".join(
            [
                f"export LD_LIBRARY_PATH={self._qnn_x86_lib}/:{self._build_folder}/lib &&",
                str(self._runner),
                *args,
            ]
        )
        logger.info("Executing on host: %s", command)
        process = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"x86 {self._runner.name} failed with exit code "
                f"{process.returncode}:\n{process.stdout}"
            )

    def pull(self, filename: str, host_path: Path) -> None:
        """Move ``filename`` from ``host_dir`` to ``host_path``."""
        shutil.move(self._require_host_dir() / filename, host_path)

    def _require_host_dir(self) -> Path:
        if self._host_dir is None:
            raise RuntimeError("setup() has not been called")
        return self._host_dir
