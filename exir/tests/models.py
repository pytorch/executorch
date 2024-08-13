# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from typing import Any, List, Optional, Tuple, Union

import executorch.exir as exir

import torch  # noqa: F401
import torch.nn as nn
from executorch.exir import to_edge
from executorch.exir.lowered_backend_module import LoweredBackendModule
from torch import Tensor
from torch.export import export

# TODO: add one more test for data dependent op plus repeat


class TensorItem(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
        h = arg1.item()
        w = arg2.item()
        torch._check(h >= 2)
        torch._check(h <= 100)
        torch._check(w >= 2)
        torch._check(w <= 100)
        return torch.ones(int(h), int(w))

    def get_random_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(10), torch.tensor(20))


class Repeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, arg1: torch.Tensor, arg2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = arg2.repeat(arg1.size(0), 1)
        return x * x, arg2 + arg2

    def get_random_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.rand(4), torch.rand(5))

    def get_dynamic_shape(self) -> Any:  # pyre-ignore[3]
        dim = torch.export.Dim("dim", max=10)
        dim2 = torch.export.Dim("dim2", max=10)
        return ({0: dim}, {0: dim2})


class ModelWithUnusedArg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
        return torch.sin(arg1)

    def get_random_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.rand(4), torch.rand(5))


class MLP(nn.Module):
    def __init__(self, n_layer: int = 1, output_size: int = 1) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.output_size = output_size
        # input shape [batch_size, n_layer+output_size]
        # each linear layer reduce the activation dim 1 size by 1.
        self.mlp = torch.nn.Sequential(
            *itertools.chain(
                *(
                    [nn.Linear(i + output_size, i - 1 + output_size)]
                    + ([nn.ReLU()] if i != 1 else [])
                    for i in range(n_layer, 0, -1)
                )
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.rand(2, self.n_layer + self.output_size),)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.clone(input)


class Reshape(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: Tensor, *new_shape: Union[torch.Size, Tuple[int, ...], List[int]]
    ) -> Tensor:
        if len(new_shape) == 1 and (
            isinstance(new_shape[0], tuple) or isinstance(new_shape[0], list)
        ):
            return x.reshape(new_shape[0])
        assert isinstance(new_shape, Union[torch.Size, Tuple[int, ...], List[int]])
        return x.reshape(new_shape)


class Transpose(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        return x.transpose(dim0, dim1)


class Mul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, other: Tensor) -> Tensor:
        # or return torch.mul(input, other)
        return input * other

    def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
        return (torch.randn(3, 2), torch.randn(3, 2))


class ElementwiseAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
        return (torch.randn(1, 3), torch.randn(1, 3))


class BasicSinMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)

    def get_random_inputs(self) -> Tuple[Tensor]:
        return (torch.randn(100),)


class CompositeDelegateModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        class DelegateAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return [x + y]

            def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
                return (torch.randn(1, 3), torch.randn(1, 3))

        delegated_m = DelegateAdd()
        edge_ir_m = to_edge(
            export(
                delegated_m,
                delegated_m.get_random_inputs(),
            )
        )
        lowered_module = LoweredBackendModule(
            edge_program=edge_ir_m.exported_program(),
            backend_id="backend_demo",
            processed_bytes=bytes("basic_module_add", encoding="utf8"),
            compile_specs=[],
        )
        self.lowered_module: LoweredBackendModule = lowered_module

    def forward(self, a: exir.Value, b: exir.Value, s: Tensor) -> Tensor:
        res = self.lowered_module(a, b)
        res = res[0] * s
        return res

    def get_random_inputs(self) -> Tuple[Tensor, Tensor, Tensor]:
        return (torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3))


class BatchMatrixMultiplication(nn.Module):
    def __init__(self, transposed: bool = False) -> None:
        super().__init__()

        # Whether the last 2 dims (-1, -2) of the input has already been
        # transposed. If yes, transpose it back before feeding to torch.bmm
        self.transposed: bool = transposed

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.transposed:
            return torch.bmm(x, y.transpose(-1, -2))
        else:
            return torch.bmm(x, y)

    def extra_repr(self) -> str:
        return f"transposed={self.transposed}"


class TensorSplit(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, sections: int, dim: int = 0) -> List[Tensor]:
        # pyre-fixme[7]: Expected `List[Tensor]` but got `Tuple[Tensor, ...]`.
        return torch.tensor_split(input, sections, dim)


class TensorSplitWithSizes(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
        # pyre-fixme[7]: Expected `List[Tensor]` but got `Tuple[Tensor, ...]`.
        return torch.split(input, split_size, dim)


class Cat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # def forward(self, tensors, dim=0):
    def forward(self, *args: Tensor, dim: int) -> Tensor:
        tensors = args[:-1]
        return torch.cat(tensors, dim)


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(input_dim)

        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout()

        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout()

    def forward(self, x: Tensor) -> Tensor:
        # LayerNorm -> Linear -> Dropout -> ReLU -> Linear -> Dropout
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y


class NoOp(nn.Module):
    """
    NoOp simply passes the input as the output.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        hidden_dim3: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ScaledDotProductAttentionModularized(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

        self.head_dim: int = embed_dim // num_heads
        self.scaling: float = self.head_dim**-0.5

        self.mul = Mul()
        self.reshape = Reshape()
        self.transpose = Transpose()
        self.bmm = BatchMatrixMultiplication(transposed=False)
        self.bmm_t = BatchMatrixMultiplication(transposed=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        # q: (L, B, D) k: (S, B, D) v: (S, B, D)
        # assert k.shape == v.shape
        # assert q.dim() == 3 and k.dim() == 3
        # assert q.size(1) == k.size(1) and q.size(2) == k.size(2)

        L, B, D = q.shape
        S = k.size(0)
        # assert D % self.head_dim == 0

        # FIXME(poweic): scaling layer!?
        # this will break the modular assumption, which makes the following
        # self.reshape to think it is using some floating inputs q because
        # id(q) is no longer the same id(q)
        # This is equiv. to `q = q * self.scaling`
        q = self.mul(q, self.scaling)

        # Reshape & transpose q from (L, B, D) to (B*H, L, D/H)
        q = self.reshape(q, (L, B * self.num_heads, self.head_dim))
        q = self.transpose(q, 0, 1)

        # Reshape & transpose k from (S, B, D) to (B*H, S, D/H)
        k = self.reshape(k, (S, B * self.num_heads, self.head_dim))
        k = self.transpose(k, 0, 1)

        # Reshape & transpose v from (S, B, D) to (B*H, S, D/H)
        v = self.reshape(v, (S, B * self.num_heads, self.head_dim))
        v = self.transpose(v, 0, 1)

        # bmm((B*H, L, D/H), (B*H, D/H, S)) -> (B*H, L, S).
        # this is equiv. to `qk = torch.bmm(q, k.transpose(-1, -2))`
        qk = self.bmm_t(q, k)
        # assert qk.shape == (B * self.num_heads, L, S)

        softmax_qk = self.softmax(qk)

        softmax_qk = self.dropout(softmax_qk)

        # bmm((B*H, L, S), (B*H, S, D/H)) -> (B*H, L, D/H).
        # this is equiv. to `attention = torch.bmm(softmax_qk, v)`
        attention = self.bmm(softmax_qk, v)
        # assert attention.shape == (B * self.num_heads, L, self.head_dim)

        # Transpose & reshape attention: (B*H, L, D/H) -> (L, B*H, D/H) -> (L, B, D).
        attention = self.transpose(attention, 0, 1)
        attention = self.reshape(attention, (L, B, self.embed_dim))

        return attention


# ------------------------------------------------------------------------------
#   Scaled Dot-Product Attention
# ------------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: Optional[float] = None,
    ) -> None:
        if embed_dim % num_heads:
            raise ValueError(
                "embed_dim ({}) must be divisible by num_heads ({})".format(
                    embed_dim, num_heads
                )
            )

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if dropout is not None and dropout > 0.0:
            self.dropout: nn.Module = nn.Dropout(p=dropout)
        else:
            self.dropout = NoOp()

        self.head_dim: int = embed_dim // num_heads
        self.scaling: float = self.head_dim**-0.5

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # q: (L, B, D) k: (S, B, D) v: (S, B, D)
        # assert k.shape == v.shape
        # assert q.dim() == 3 and k.dim() == 3
        # assert q.size(1) == k.size(1) and q.size(2) == k.size(2)

        L, B, D = q.shape
        S = k.size(0)
        # assert D % self.head_dim == 0

        q = q * self.scaling
        q = q.reshape(L, B * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # (B*H, L, D/H)

        k = k.reshape(S, B * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # (B*H, S, D/H)

        v = v.reshape(S, B * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # (B*H, S, D/H)

        # bmm((B*H, L, D/H), (B*H, D/H, S)) -> (B*H, L, S).
        qk = torch.bmm(q, k.transpose(1, 2))
        # assert qk.shape == (B * self.num_heads, L, S)

        # TODO(cfyeh): figure out if we really need input to be float.
        softmax_qk = nn.functional.softmax(qk.float(), dim=-1)

        # softmax_qk = self.dropout(softmax_qk)

        # bmm((B*H, L, S), (B*H, S, D/H)) -> (B*H, L, D/H).
        attention = torch.bmm(softmax_qk, v)
        # assert attention.shape == (B * self.num_heads, L, self.head_dim)

        # (B*H, L, D/H) -> (L, B*H, D/H) -> (L, B, D).
        attention = attention.transpose(0, 1).reshape(L, B, self.embed_dim)

        return attention


class Emformer(nn.Module):
    def __init__(
        self,
        l_dim: int = 32,
        m_dim: int = 8,
        c_dim: int = 8,
        r_dim: int = 8,
        input_dim: int = 256,
        ffn_hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.l_dim = l_dim
        self.m_dim = m_dim
        self.c_dim = c_dim
        self.r_dim = r_dim

        self.input_dim = input_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.split = TensorSplit()
        self.elem_add = ElementwiseAdd()

        self.attn = ScaledDotProductAttention(
            embed_dim=input_dim,
            num_heads=8,
        )

        self.ffn = FeedForwardBlock(input_dim, ffn_hidden_dim)

        self.layer_norm = nn.LayerNorm(input_dim)

        self.linear_k = nn.Linear(self.input_dim, self.input_dim)
        self.linear_v = nn.Linear(self.input_dim, self.input_dim)
        self.linear_q = nn.Linear(self.input_dim, self.input_dim)

    def get_random_inputs(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        inputs = (
            torch.randn(self.m_dim, 1, self.input_dim),
            torch.randn(self.c_dim, 1, self.input_dim),
            torch.randn(self.r_dim, 1, self.input_dim),
            torch.randn(self.l_dim, 1, self.input_dim),
            torch.randn(self.l_dim, 1, self.input_dim),
        )
        return inputs

    def forward(
        self, M: Tensor, C: Tensor, R: Tensor, K_L: Tensor, V_L: Tensor
    ) -> Tensor:
        """
        The Emformer block takes [M_i^n, C_i^n, R_i^n] and [K_{L,i}^n, V_{L,i}^n]
        as inputs and outputs [C_i^{n+1}, R_i^{n+1}].
        See Fig. 1(b) Emformer and equations 6, 7, 8 - 13 in the original paper
        https://arxiv.org/pdf/2010.10759.pdf

        Ex:
         - self.input_dim =
         - L.shape = 30 x 1 x 512
         - M.shape =  2 x 1 x 512
         - C.shape =  5 x 1 x 512
         - R.shape =  1 x 1 x 512
        """
        # Equation 8
        CR = torch.cat([C, R], 0)
        CR_normed = self.layer_norm(CR)
        # C_normed = self.layer_norm(C)
        # R_normed = self.layer_norm(R)

        # Equation 9 and 10
        if True:
            MCR = torch.cat([M, C, R], 0)
            K_MCR = self.linear_k(MCR)
            V_MCR = self.linear_v(MCR)

            K_M, K_C, K_R = self.split(K_MCR, 3)
            V_M, V_C, V_R = self.split(V_MCR, 3)
        else:
            K_M, K_C, K_R = self.linear_k(M), self.linear_k(C), self.linear_k(R)
            V_M, V_C, V_R = self.linear_v(M), self.linear_v(C), self.linear_v(R)

        K = torch.cat([K_M, K_L, K_C, K_R], 0)
        V = torch.cat([V_M, V_L, V_C, V_R], 0)

        # Equation 11 and 12
        Q_CR = self.linear_q(CR_normed)
        Z_CR = self.attn(Q_CR, K, V)
        Z_CR = self.elem_add(Z_CR, CR)
        # Q_C = self.linear_q(C_normed)
        # Q_R = self.linear_q(R_normed)
        # Z_C = self.attn(Q_C, K, V)
        # Z_R = self.attn(Q_R, K, V)
        # Z_C = self.elem_add(Z_C, C)
        # Z_R = self.elem_add(Z_R, R)

        # Equation 6
        Z_CR_normed = self.layer_norm(Z_CR)
        ffn_out = self.ffn(Z_CR_normed)

        # Equation 7
        output = self.layer_norm(self.elem_add(ffn_out, Z_CR))

        # m = self.attn(

        return output


# List of models that we want to export
# TODO(angelayi): enable ControlFlowWhile test once we enable functionalization
MODELS = [
    ["basic_sin_max", BasicSinMax()],
    ["composite_delegate", CompositeDelegateModule()],
]
