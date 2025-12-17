import torch.nn.functional as F

from executorch.examples.models.llama.lora import LoRALinear
from executorch.examples.models.llama.model_args import ModelArgs
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LoRAFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, args: ModelArgs):
        super().__init__()

        if args.r is None or args.lora_alpha is None:
            raise ValueError(
                "LoRA rank and alpha must be specified for LoRAFeedForward."
            )

        self.w1 = (
            LoRALinear(
                in_dim=dim,
                out_dim=hidden_dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=False,
            )
            if "gate_proj" in args.target_modules
            else nn.Linear(dim, hidden_dim, bias=False)
        )

        self.w2 = (
            LoRALinear(
                in_dim=hidden_dim,
                out_dim=dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=False,
            )
            if "down_proj" in args.target_modules
            else nn.Linear(hidden_dim, dim, bias=False)
        )

        self.w3 = (
            LoRALinear(
                in_dim=dim,
                out_dim=hidden_dim,
                rank=args.r,
                alpha=args.lora_alpha,
                dropout=0.0,
                use_bias=False,
            )
            if "up_proj" in args.target_modules
            else nn.Linear(dim, hidden_dim, bias=False)
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
