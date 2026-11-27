# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an inference runner on a target device through ``Device``."""

from __future__ import annotations

import getpass
import logging
import shlex
from pathlib import Path
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.qualcomm.export_utils import Device, QnnConfig

logger = logging.getLogger(__name__)


class DevicePlatform:
    """Runs a runner on a target device through ``Device``.

    The target is an Android/OE-Linux device over adb. ``Device`` also bridges
    to a local Windows-on-Snapdragon target, which this platform does not
    serve: ``execute`` builds a POSIX shell command and the default workspace
    is a POSIX path.

    Its responsibility is the device session: ``setup`` clears ``workspace`` and
    pushes the runner, the backend libraries, and the files into it, and
    ``add_file`` pushes one more file for a request. The runner reads them there
    by name, so ``runner_path`` and ``runner_input_list`` return bare names. It
    writes its outputs to the session's output folder on the device, from which
    ``pull`` copies them back. ``execute`` runs the runner from the workspace;
    ``Device`` raises if the command fails.

    Device PTEs are compiled with shared buffers, so every runner command gets
    ``--shared_buffer``; generation is not limited.

    Args:
        qnn_config: Device, build-folder, and backend settings.
        workspace: Folder on the device holding the session's files. Defaults
            to a per-user folder, as the legacy flow does: ``setup`` clears the
            workspace, so a shared one would wipe another user's or CI job's
            files mid-run.
    """

    # Device PTEs are always compiled with shared buffers; see compile_spec_builder.
    target_flags: Tuple[str, ...] = ("--shared_buffer",)
    limits_generation: bool = False

    def __init__(
        self,
        qnn_config: "QnnConfig",
        workspace: Optional[str] = None,
    ) -> None:
        self._qnn_config = qnn_config
        self._workspace = (
            workspace or f"/data/local/tmp/{getpass.getuser()}/genai_pipeline"
        )
        self._device: Optional["Device"] = None
        self._runner_name: Optional[str] = None

    def setup(self, runner: str, files: Sequence[Path], host_dir: Path) -> None:
        """Start a device session and push the runner and ``files`` to the device.

        Args:
            runner: The runner binary, relative to the build folder.
            files: Host files the runner reads, PTEs included.
            host_dir: Unused; outputs stay on the device until ``pull``.
        """
        from executorch.backends.qualcomm.export_utils import Device

        # PTEs travel with ``files`` rather than as Device's ``pte_path``,
        # which every later push would upload again.
        self._device = Device(
            qnn_config=self._qnn_config,
            pte_path=[],
            workspace=self._workspace,
            runner=runner,
        )
        self._runner_name = Path(runner).name
        self._device.push(inputs=None, files=[str(path) for path in files])

    def add_file(self, path: Path) -> None:
        """Push one more file into the session's workspace.

        Through the bridge, since ``Device.push`` would upload the runner again.
        """
        self._require_device().bridge.push_file(str(path), self._workspace)

    def runner_path(self, host_path: Path) -> str:
        """Return the file's name, which the runner resolves in the workspace."""
        return host_path.name

    def runner_input_list(self, host_path: Path) -> str:
        """Return the list's name: its entries are pushed beside it in the workspace."""
        return host_path.name

    def output_dir(self) -> str:
        """Return the session's output folder on the device."""
        return self._require_device().output_folder

    def execute(self, args: Sequence[str]) -> None:
        """Run the runner from the workspace; ``Device`` raises on failure."""
        device = self._require_device()
        command = " ".join(
            [f"cd {shlex.quote(self._workspace)} &&", f"./{self._runner_name}", *args]
        )
        logger.info("Executing on device: %s", command)
        device.execute(custom_runner_cmd=command)

    def pull(self, filename: str, host_path: Path) -> None:
        """Copy ``filename`` from the device output folder to ``host_path``."""
        device = self._require_device()
        device.pull(
            host_output_path=str(host_path),
            device_output_path=f"{device.output_folder}/{filename}",
        )

    def _require_device(self) -> "Device":
        if self._device is None:
            raise RuntimeError("setup() has not been called")
        return self._device
