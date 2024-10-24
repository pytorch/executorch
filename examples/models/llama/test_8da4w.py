import os

import torch.cuda

from torch import nn
from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear


class Attention(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.wq = Int8DynActInt4WeightLinear(
            in_features=2048,
            out_features=2048,
            bias=False,
            device=device,
            groupsize=32,
            precision=torch.float32,
            scales_precision=torch.float32,
        )

    def forward(self, x: torch.tensor):
        return self.wq.forward(x)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.load(f"{os.path.dirname(__file__)}/x.pt").to(device=device)
    checkpoint = torch.load(
        f"{os.path.dirname(__file__)}/wq.pth",
        map_location=device,
        mmap=True,
    )
    print(f"input {input}")
    results = []
    iterations = 10
    for i in range(iterations):
        model = Attention(device).to(device=device)
        model.load_state_dict(checkpoint, strict=False, assign=True)

        result = model.forward(input)
        exist = False
        for existing_result in results:
            if torch.allclose(result, existing_result):
                exist = True
                break
        if not exist:
            results.append(result)
    print(f"Generated {len(results)} results with {iterations} iterations")
    for i, result in enumerate(results):
        print(f"result {i} {result}")


if __name__ == "__main__":
    main()
