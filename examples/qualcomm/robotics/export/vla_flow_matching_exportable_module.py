import torch
from torch import nn

from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    ACTION_EXPERT_GRAPH_NAME,
    PREFIX_MODEL_GRAPH_NAME,
    VISION_ENCODER_GRAPH_NAME,
)


class VLAFlowMatchingExportableModule(nn.Module):
    """Own eager and quantized components and component exports."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.exported_modules: dict[str, torch.fx.GraphModule] = {}

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(*inputs)

    def export(
        self,
    ) -> None:
        """Export the fixed eager Vision, Prefix, and Action Expert components."""
        # strict=False, as the Action Expert takes the prefix K/V as lists of tensors.
        for name, component in (
            (VISION_ENCODER_GRAPH_NAME, self.model.vision_encoder),
            (PREFIX_MODEL_GRAPH_NAME, self.model.prefix_model),
            (ACTION_EXPERT_GRAPH_NAME, self.model.action_expert),
        ):
            self.exported_modules[name] = torch.export.export(
                component,
                component.get_example_inputs(),
                strict=False,
            ).module()
