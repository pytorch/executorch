from kokoro import KModel
import torch
from torch import nn
from torch.export import default_decompositions, Dim, export_for_training

torch.manual_seed(42)

class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        return self.model.forward_with_tokens(
            input_ids,
            ref_s,
            speed,
        )

repo_id = "hexgrad/Kokoro-82M"
model = KModel(repo_id=repo_id).eval()
wrapped_model = WrappedModel(model)

input_ids = torch.randint(1, 100, (48,))
input_ids = torch.LongTensor([[0, *input_ids, 0]])  # S = [1, 50]
style = torch.randn(1, 256)
speed = torch.randint(1, 10, (1,)).int()
example_inputs = (input_ids, style, speed)

"""
Original model output is:
(tensor([-0.1578,  0.0960,  0.0831,  ...,  0.1224, -0.0831,  0.1492]), tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2]))
"""
print(wrapped_model(*example_inputs))


dynamic_shapes = {
    # "input_ids": {0: 1, 1: Dim("input_ids", min=2, max=100)},
    "input_ids": {},
    "ref_s": {},
    "speed":{},
}

exported_program = export_for_training(wrapped_model, args=example_inputs, dynamic_shapes=dynamic_shapes, strict=True)
exported_program = exported_program.run_decompositions(default_decompositions())
exported_program.run_decompositions()
