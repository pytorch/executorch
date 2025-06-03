# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Any, Dict, Tuple

import torch
from torch.fx import GraphModule
from torch.fx.interpreter import Interpreter


class IntermediateOutputCapturer(Interpreter):
    def __init__(self, module: GraphModule):
        super().__init__(module)

    def run_and_capture(self, *args, **kwargs) -> Dict[Tuple[int, ...], Any]:
        captured_outputs = {}

        def capture_run_node(n: torch.fx.Node) -> Any:
            result = super(IntermediateOutputCapturer, self).run_node(n)
            debug_handle = n.meta.get("debug_handle", None)
            if debug_handle is not None and n.op == "call_function":
                # Convert the debug handle to a tuple to use as a dictionary key
                key = (
                    (debug_handle,)
                    if isinstance(debug_handle, int)
                    else tuple(debug_handle)
                )
                # Handle tensor results by detaching and cloning
                if isinstance(result, torch.Tensor):
                    captured_outputs[key] = result.detach().clone()
                elif isinstance(result, (tuple, list)):
                    captured_outputs[key] = [
                        r.detach().clone() if isinstance(r, torch.Tensor) else r
                        for r in result
                    ]
                else:
                    captured_outputs[key] = result
            return result

        original_run_node = self.run_node
        self.run_node = capture_run_node
        self.run(*args, **kwargs)
        self.run_node = original_run_node
        return captured_outputs
