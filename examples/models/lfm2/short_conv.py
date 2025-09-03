import torch
from executorch.examples.models.llama.attention import ForwardOptions
from executorch.examples.models.llama.feed_forward import FeedForward

from executorch.examples.models.llama.norm import RMSNorm
from torch import nn


class ShortConv(nn.Module):
    def __init__(
        self,
        dim: int,
        L_cache: int = 3,
        bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.dim = dim
        self.L_cache = L_cache
        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=L_cache,
            padding=0,  ## we don't need padding since we handle it manually
            groups=dim,
            bias=bias,
        )

        conv_state = torch.zeros(
            1,  ## batch size is assumed to be 1 for now
            dim,
            L_cache - 1,
            device="cpu",
        )
        self.register_buffer("conv_state", conv_state)

        ## better performance in Executorch with separate projections
        self.B_proj = nn.Linear(dim, dim, bias=bias)
        self.C_proj = nn.Linear(dim, dim, bias=bias)
        self.x_proj = nn.Linear(dim, dim, bias=bias)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, dim = x.size()
        assert batch_size == 1, "batch_size must be 1"

        B = self.B_proj(x).transpose(-1, -2)  # (batch_size, dim, seq_len)
        C = self.C_proj(x).transpose(-1, -2)  # (batch_size, dim, seq_len)
        x = self.x_proj(x).transpose(-1, -2)  # (batch_size, dim, seq_len)

        Bx = B * x  # (batch_size, dim, seq_len)

        ## This is where we handle padding
        ## By default, the conv_state is initialized to 0.
        #  So, assuming prefill is done on an empty cache, concatenating conv_state to the beginning of the sequence acts similary to
        ## using nn.Conv1d(padding=L_cache-1) (for prefill) without no manual padding.
        ## However, the manual padding has the added benefit of being correct during decode, when the cache is not initialized to 0.
        Bx = torch.cat(
            [self.conv_state, Bx], dim=-1
        )  # (batch_size, dim, seq_len + L_cache - 1)

        ## Update the conv_state
        new_conv_state = Bx[
            ..., -(self.L_cache - 1) :
        ]  # (batch_size, dim, L_cache - 1)
        with torch.no_grad():
            self.conv_state.copy_(new_conv_state)

        conv_out = self.conv(Bx)[..., : x.size(-1)]  # (batch_size, dim, seq_len)
        y = C * conv_out  # (batch_size, dim, seq_len)

        y = y.transpose(-1, -2)  # (batch_size, seq_len, dim)
        y = y.contiguous()  # (batch_size, seq_len, dim)
        y = self.out_proj(y)  # (batch_size, seq_len, dim)
        return y

    def reset_cache(self):
        self.conv_state.zero_()


class ShortConvBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.L_cache = 3  # hardcode 3 for now
        self.conv = ShortConv(dim, self.L_cache, bias=False)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        # use attention_norm norm instead of operator_norm to unify with TransformerBlock
        self.attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x,
        freqs_cos=None,
        freqs_sin=None,
        _unused_attn_options: ForwardOptions = None,
    ):  # x: 1xN
        h = self.conv.forward(self.attention_norm(x))
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, None

    def reset_cache(self):
        self.conv.reset_cache()
