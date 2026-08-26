# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Carrying externalized submodules from export time into ``preprocess``.

``externalize_modules`` must run before ``torch.export``, while the
``nn.Module`` still exists, but its result is only needed much later, when the
backend converts a partitioned subgraph. The result holds live
``ExportedProgram`` objects, so it cannot travel in a ``CompileSpec``, and node
metadata does not survive edge lowering.

coreai-torch returns one module per *call site*, and a submodule invoked more
than once (a shared norm, say) gives several that agree on ``op_name`` and
differ in ``name``. Entries are therefore keyed by ``name``, the per-call-site
identity, and grouped by ``op_name`` on the way out so ``preprocess`` gets every
call site of the ops its own subgraph invokes.

Nothing is serialized: like the sidecar output directory, this is build-time
state with no meaning to the runtime. References are weak, so an entry lives
exactly as long as the caller's own reference to what ``externalize_modules``
returned, and a long-lived process lowering many models does not accumulate
their submodule programs.
"""

import weakref
from typing import List, Sequence

from coreai_torch.externalize import _ExternalizedExportedProgram as ExternalizedModule

_PREPARED: "weakref.WeakValueDictionary[str, ExternalizedModule]" = (
    weakref.WeakValueDictionary()
)


def register(modules: Sequence[ExternalizedModule]) -> None:
    """Make prepared submodules available to ``preprocess``."""
    for module in modules:
        _PREPARED[module.name] = module


def lookup(op_names: Sequence[str]) -> List[ExternalizedModule]:
    """Every prepared call site of the given ops, in registration order.

    Repeats in ``op_names`` are ignored: the ops are read off graph nodes, so
    an op invoked at several call sites is named several times, while each of
    its modules must be handed over exactly once.

    Raises:
        KeyError: If an op has no prepared submodule. Either the partitioner
            was not given them, the caller stopped holding the result of
            ``externalize_modules`` before lowering finished, or partitioning
            and preprocessing ran in different processes, which cannot work:
            the payload holds live ExportedProgram objects.
    """
    wanted = set(op_names)
    # Resolve in one pass: the values are weakly held, so a membership test
    # followed by a lookup could see an entry collected in between.
    found: List[ExternalizedModule] = [
        module for module in list(_PREPARED.values()) if module.op_name in wanted
    ]
    missing = wanted - {module.op_name for module in found}
    if missing:
        raise KeyError(
            f"no prepared submodule registered for {sorted(missing)}. Pass the "
            f"result of externalize_modules to CoreAIPartitioner via "
            f"externalized_modules=, keep a reference to it until lowering "
            f"finishes, and lower in the same process."
        )
    return found


def clear() -> None:
    """Drop all registered submodules. For tests."""
    _PREPARED.clear()
