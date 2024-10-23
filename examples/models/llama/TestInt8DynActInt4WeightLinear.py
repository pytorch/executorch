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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.load("/home/lunwenh/models/x.pt").to(device=device)
    checkpoint = torch.load(
        "/home/lunwenh/models/wq.pth",
        map_location=device,
        mmap=True,
    )
    print(f"input {input}")
    for i in range(5):
        model = Attention(device).to(device=device)
        model.load_state_dict(checkpoint, strict=False, assign=True)

        print(model.forward(input))


if __name__ == "__main__":
    main()
