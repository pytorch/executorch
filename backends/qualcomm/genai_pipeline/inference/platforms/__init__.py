# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where an inference runner runs: on an Android device or on the x86 host.

A ``RunnerPlatform`` owns how the runner's files reach it and how the runner is
launched; it knows nothing about the model. The runner adapters own what to run
and how to read its outputs, and drive a platform injected at construction.

``DefaultDeviceRunnerAdapter`` deliberately stays outside this abstraction: it
runs the generic ``qnn_executor_runner`` through ``Device``'s own default
command, which a platform does not model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable, Sequence, Tuple, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.inference.platforms.device import (
    DevicePlatform,
)
from executorch.backends.qualcomm.genai_pipeline.inference.platforms.x86 import (
    X86Platform,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.export_utils import QnnConfig


@runtime_checkable
class RunnerPlatform(Protocol):
    """Where a runner runs: how its files reach it and how it is launched.

    A platform owns everything that differs between running on an Android
    device and on the x86 host, and only that:

    - **Getting files to the runner.** It deploys the runner binary, PTEs, and
      runtime files, and later one request's input files.
    - **Naming them for the runner.** It says how the runner refers to a host
      file or input list, and where the runner writes its outputs.
    - **Launching the runner** and bringing its outputs back.
    - **Target facts** the runner flags depend on, as ``target_flags`` and
      ``limits_generation``.

    It knows nothing about the model: which runner, which PTEs, which flags, and
    how to decode the outputs are the runner adapter's. The CLI picks the
    platform with ``get_runner_platform`` and injects it into the adapter, which
    drives it from its own lifecycle stages::

        adapter.push_artifacts  ->  setup, runner_input_list
        adapter.push_inputs     ->  add_file
        adapter.execute         ->  runner_path, output_dir, target_flags,
                                    then execute
        adapter.pull_results    ->  pull

    The inference strategy and the evaluators never see a platform.

    ``setup`` starts a session and must be called first; the other methods that
    depend on the session raise ``RuntimeError`` before it.

    Attributes:
        target_flags: Runner flags this target always passes.
        limits_generation: Whether generation should be kept short because
            the target runs slowly, as the x86 emulator does. How short is the
            adapter's choice, since it depends on the model.
    """

    target_flags: Tuple[str, ...]
    limits_generation: bool

    def setup(self, runner: str, files: Sequence[Path], host_dir: Path) -> None:
        """Start a session in which the runner can read ``files``.

        Args:
            runner: The runner binary, relative to the build folder.
            files: Host files the runner reads, PTEs included.
            host_dir: Host directory the session may use for runner I/O. A
                platform that runs the runner on the host writes its outputs
                here; one that copies them back elsewhere need not use it.
        """
        ...

    def add_file(self, path: Path) -> None:
        """Make one more host file readable by the runner, e.g. a token file."""
        ...

    def runner_path(self, host_path: Path) -> str:
        """Return how the runner refers to a file provided from ``host_path``."""
        ...

    def runner_input_list(self, host_path: Path) -> str:
        """Return how the runner refers to an input list provided from ``host_path``.

        An input list names the input files it lists by file name, relative to
        the list's own directory, as ``generate_inputs`` writes it. The platform
        makes those entries resolvable where the runner reads them.
        """
        ...

    def output_dir(self) -> str:
        """Return the directory the runner writes its outputs to."""
        ...

    def execute(self, args: Sequence[str]) -> None:
        """Launch the runner with ``args``; raise if it fails."""
        ...

    def pull(self, filename: str, host_path: Path) -> None:
        """Bring the runner output ``filename`` back to ``host_path``."""
        ...


def get_runner_platform(qnn_config: "QnnConfig", enable_x86_64: bool) -> RunnerPlatform:
    """Select where the runner runs from the CLI's target flag.

    Args:
        qnn_config: Device, build-folder, and backend settings.
        enable_x86_64: Whether to run the x86 build of the runner on the host.

    Returns:
        An ``X86Platform`` when ``enable_x86_64`` is set, else a
        ``DevicePlatform``.
    """
    if enable_x86_64:
        return X86Platform(qnn_config)
    return DevicePlatform(qnn_config)


__all__ = [
    "DevicePlatform",
    "get_runner_platform",
    "RunnerPlatform",
    "X86Platform",
]
