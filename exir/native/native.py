# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Direct native lowering.

Where the ExecuTorch flow is export -> [quantize] -> to_edge -> to_backend -> to_executorch, this
is export -> [quantize] -> to_native, and yields a NativeProgramManager to save as a .ptn. See README.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from torch.export import ExportedProgram


class NativeProgramManager:
    """One or more methods lowered for the native runtime, ready to save.

    The counterpart to ExecutorchProgramManager: ``methods`` and ``save`` mirror it
    deliberately, so the two pipelines read alike. The surface is intentionally
    small, we will expand this later.

    The graph blob is the only source of truth: method names and package-wide
    mutability are read back out of it on demand rather than carried alongside,
    so a manager cannot describe a package it does not hold.

    Constants are held as tensor references returned by lowering. ``save`` may
    normalize device or layout before writing them.
    """

    def __init__(
        self,
        ptg: bytes,
        constants: dict[str, torch.Tensor],
    ) -> None:
        # TODO add args help
        raise NotImplementedError

    @property
    def methods(self) -> set[str]:
        """Names of the methods in the program."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Write the program and its constants to ``path`` as a .ptn package."""
        raise NotImplementedError


def to_native(
    programs: "ExportedProgram | dict[str, ExportedProgram]",
) -> NativeProgramManager:
    """Lower ExportedProgram(s) through the native backend.

    ``programs`` is a single ``ExportedProgram`` (lowered as the sole ``forward``
    method) or a dict mapping method name to ``ExportedProgram``.

    Nothing about lowering is configurable yet. The passes, the partitioner, and
    the edge compile config are all owned here, because correctness of the package
    depends on them. Exposing the compile config is worth reconsidering.
    ``constant_methods`` and ETRecord are likewise deferred.

    Args:
        programs: The exported program(s) to lower.

    Returns:
        A ``NativeProgramManager``; call ``save`` to write a .ptn package.

    Raises:
        TypeError: If ``programs`` is not an ``ExportedProgram`` or a method-name
            dictionary containing only ``ExportedProgram`` values.
        ValueError: If the method dictionary is empty, a method name is empty, a
            method does not fully delegate to the native backend, a constant
            differs between methods, or a data key the graph references has no
            backing tensor.
    """
    raise NotImplementedError


__all__ = ["NativeProgramManager", "to_native"]
