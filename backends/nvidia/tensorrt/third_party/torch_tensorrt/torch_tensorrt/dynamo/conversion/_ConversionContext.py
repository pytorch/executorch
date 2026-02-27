from dataclasses import dataclass, field

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.types import TRTNetwork


@dataclass
class ConversionContext:
    """Class representing the context for conversion of a particular network

    Args:
        net: TensorRT Network being built
        compilation_settings: Settings selected by the user for compilation
        requires_output_allocator: Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
        weight_refit_map: Dictionary mapping weight names to their corresponding np.array
        cpu_weights_reference_holder: Dictionary mapping weight names to their corresponding torch.Tensor
    """

    net: TRTNetwork
    compilation_settings: CompilationSettings = field(
        default_factory=CompilationSettings
    )
    requires_output_allocator: bool = False
    weight_refit_map: dict[str, torch.Tensor] = field(default_factory=dict)
    cpu_weights_reference_holder: list[torch.Tensor] = field(default_factory=list)

    def record_weight(self, name: str, weight: torch.Tensor) -> None:
        """
        Record the weight and name for refitting and CPU reference.
        For the refit map, the key is the weight name that appears in the TRT engine and the value is the weight tensor.
        For the CPU reference holder, we need to hold the reference to the weight tensor until the whole compilation process is complete.

        Args:
            name: Name of the weight
            weight: Weight to record
        """
        self.weight_refit_map[name] = weight
        self.cpu_weights_reference_holder.append(weight)

    def clear_cpu_weights_reference_holder(self) -> None:
        self.cpu_weights_reference_holder.clear()
