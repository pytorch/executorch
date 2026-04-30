import logging
from typing import Any, Union

import torch
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)

logger = logging.getLogger(__name__)


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(
        self, module: Union[torch.fx.GraphModule, CudaGraphsTorchTensorRTModule]
    ) -> None:
        rt_mods = []
        self.current_device_budget = 0
        self.cuda_graphs_module = None

        if isinstance(module, CudaGraphsTorchTensorRTModule):
            self.cuda_graphs_module = module
            module = module.compiled_module
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and isinstance(
                rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                rt_mods.append((name, rt_mod))
                self.current_device_budget += rt_mod.get_device_memory_budget()
        self.streamable_budget = [
            mod.get_streamable_device_memory_budget() for _, mod in rt_mods
        ]
        self.rt_mods = rt_mods
        total_device_budget = sum(self.streamable_budget)
        super().__setattr__("device_budget", self.current_device_budget)
        super().__setattr__("total_device_budget", total_device_budget)

    def get_automatic_weight_streaming_budget(self) -> int:
        ws_budget_bytes = 0
        for _, rt_mod in self.rt_mods:
            ws_budget_bytes += rt_mod.get_automatic_device_memory_budget()
        return ws_budget_bytes

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        if self.total_device_budget > 0:
            logger.debug(
                f"Revert weight streaming budget to initial size: {self.current_device_budget}"
            )
            self.device_budget = self.current_device_budget

    def _set_streamable_weight_bytes(self, requested_budget: int) -> int:
        ws_budget_bytes = 0
        total_bytes = self.total_device_budget
        if total_bytes == 0:
            raise RuntimeError(
                "Streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
            )
        elif total_bytes < requested_budget:
            logger.error(
                f"Requested budget is greater than streamable bytes: {total_bytes}. requested budget: {requested_budget}"
            )
            requested_budget = total_bytes
        elif requested_budget < 0:
            raise RuntimeError("Requested budget cannot be negative")
        # Normalized size is applied for multiple runtime module.
        # e.g. 100B budget is applied to two modules and they have 1000B and 3000B streamable size respectively.
        # Then 25B and 75B are applied for each module.
        normalized_size = [
            int(streamable_bytes / total_bytes * requested_budget)
            for streamable_bytes in self.streamable_budget
        ]
        if self.cuda_graphs_module:
            self.cuda_graphs_module.is_weight_streaming_set = True
            self.cuda_graphs_module._reset_captured_graph()

        for i, (name, rt_mod) in enumerate(self.rt_mods):
            rt_mod._reset_captured_graph()
            ws_budget_bytes += rt_mod.set_device_memory_budget(normalized_size[i])
            logger.debug(f"Set weight streaming size {normalized_size[i]} for {name}")

        return ws_budget_bytes

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "device_budget":
            value = self._set_streamable_weight_bytes(value)
        super().__setattr__(name, value)


def weight_streaming(
    module: torch.fx.GraphModule,
) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
