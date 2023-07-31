# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401

from executorch.exir.schema import Frame, FrameList

# pyre-fixme[21]: Could not find module `executorch.extension.pybindings.core`.
from executorch.extension.pybindings.core import Module


# pyre-fixme[11]: Annotation `Module` is not defined as a type.
def _stacktraces(module: Module, execution_plan_idx: int, chain_idx: int):
    chain = module.program().execution_plan(execution_plan_idx).chain(chain_idx)
    stacktraces = chain.stacktraces()

    if stacktraces is None:
        return None

    result = []
    for stacktrace in stacktraces:
        frame_list = [
            Frame(
                filename=filename,
                lineno=lineno,
                name=name,
                context=context,
            )
            for (filename, lineno, name, context) in stacktrace
        ]
        result.append(FrameList(frame_list))
    return result
