# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Carrying externalized submodules from export time into ``preprocess``.

``coreai_torch.externalize_modules`` must run before ``torch.export``, while the
``nn.Module`` still exists, but its result is only needed much later, when the
backend converts a partitioned subgraph. The result holds live
``ExportedProgram`` objects, so it cannot travel in a ``CompileSpec``, and node
metadata does not survive edge lowering.

Entries are keyed by op name, which coreai-torch makes unique per
externalization, so ``preprocess`` can look up exactly the submodules its own
subgraph calls. Nothing is serialized: like the sidecar output directory, this
is build-time-only state and has no meaning to the runtime.

References are weak, so an entry lives exactly as long as the caller's own
reference to what ``externalize_modules`` returned. A long-lived process
lowering many models does not accumulate their submodule programs.
"""

import weakref
from typing import List, Sequence

from coreai_torch import ExternalizedModule

_PREPARED: "weakref.WeakValueDictionary[str, ExternalizedModule]" = (
    weakref.WeakValueDictionary()
)


def register(modules: Sequence[ExternalizedModule]) -> None:
    """Make prepared submodules available to ``preprocess``."""
    for module in modules:
        _PREPARED[module.op_name] = module


def lookup(op_names: Sequence[str]) -> List[ExternalizedModule]:
    """Return the prepared submodules for the given op names.

    Raises:
        KeyError: If an op has no prepared submodule. Either the partitioner
            was not given them, the caller stopped holding the result of
            ``externalize_modules`` before lowering finished, or partitioning
            and preprocessing ran in different processes, which cannot work:
            the payload holds live ExportedProgram objects.
    """
    missing = [name for name in op_names if name not in _PREPARED]
    if missing:
        raise KeyError(
            f"no prepared submodule registered for {sorted(missing)}. Pass the "
            f"result of coreai_torch.externalize_modules to CoreAIPartitioner "
            f"via externalized_modules=, keep a reference to it until lowering "
            f"finishes, and lower in the same process."
        )
    return [_PREPARED[name] for name in op_names]


def clear() -> None:
    """Drop all registered submodules. For tests."""
    _PREPARED.clear()
