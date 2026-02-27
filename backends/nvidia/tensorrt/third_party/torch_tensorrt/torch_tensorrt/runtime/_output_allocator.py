import logging
from typing import Any, Union

import torch
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)

logger = logging.getLogger(__name__)


class _OutputAllocatorContextManager(object):
    """
    Helper class to set up output_allocator
    """

    def __init__(
        self, module: Union[torch.fx.GraphModule, CudaGraphsTorchTensorRTModule]
    ) -> None:
        if isinstance(module, CudaGraphsTorchTensorRTModule):
            rt_mods = [module]
        else:
            rt_mods = []

            for name, rt_mod in module.named_children():
                if "_run_on_acc" in name and isinstance(
                    rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
                ):
                    rt_mods.append(rt_mod)

        self.rt_mods = rt_mods

    def set_output_allocator_output(self, enable: bool) -> None:
        for mod in self.rt_mods:
            mod.set_use_output_allocator(enable)

    def __enter__(self) -> "_OutputAllocatorContextManager":
        # Enable output_allocator for TRT submodules
        self.set_output_allocator_output(True)
        return self

    def __exit__(self, *args: Any) -> None:
        # Disable output_allocator
        self.set_output_allocator_output(False)


def enable_output_allocator(
    module: torch.fx.GraphModule,
) -> _OutputAllocatorContextManager:
    return _OutputAllocatorContextManager(module)
