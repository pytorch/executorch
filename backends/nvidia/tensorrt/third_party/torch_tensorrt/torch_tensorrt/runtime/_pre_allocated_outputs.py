import logging
from typing import Any

import torch
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


class _PreAllocatedOutputContextManager(object):
    """
    Helper class used to enable pre-allocated output feature in runtime module
    """

    def __init__(self, module: torch.fx.GraphModule) -> None:
        rt_mods = []
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and isinstance(
                rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                rt_mods.append(rt_mod)
        self.rt_mods = rt_mods

    def set_pre_allocated_output(self, enable: bool) -> None:
        for mod in self.rt_mods:
            mod.set_pre_allocated_outputs(enable)

    def __enter__(self) -> "_PreAllocatedOutputContextManager":
        # Enable pre-allocated output
        self.set_pre_allocated_output(True)
        return self

    def __exit__(self, *args: Any) -> None:
        # Disable pre-allocated output
        self.set_pre_allocated_output(False)


def enable_pre_allocated_outputs(
    module: torch.fx.GraphModule,
) -> _PreAllocatedOutputContextManager:
    return _PreAllocatedOutputContextManager(module)
