import torch.cuda
from torchao.quantization.GPTQ import _check_linear_int4_k, Int8DynActInt4WeightLinear

from torch import nn
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
            scales_precision=torch.float32
        )

    def forward(self, x: torch.tensor):
        return self.wq.forward(x)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.load("file/to/input/tensor", map_location=device)
    checkpoint = torch.load("/Users/lunwenh/models/1B_spin_new_format/consolidated.00.pth", map_location=device,
                            mmap=True)
    for i in range(5):
        model = Attention(device)
        model.load_state_dict(checkpoint, strict=False, assign=True)

        print(model.forward(input))

if __name__ == "__main__":
    main()