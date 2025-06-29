from typing import Optional, Tuple

import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.library.custom_op("coreml::sdpa", mutates_args=())
def sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Same as F.scaled_dot_product_attention, but with custom op to avoid lowering during dialect conversion."""
    return torch.ops.aten.scaled_dot_product_attention.default(
        q, k, v, attn_mask=attn_mask
    )


@torch.library.register_fake("coreml::sdpa")
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Fake implementation with the right output shape, which is required for torch.compile/export/fx tracing."""
    expected_shape = list(q.shape)
    expected_shape[-1] = v.shape[-1]
    return q.new_empty(expected_shape)


def remove_graph_asserts(exported_program):
    assert_functions = [
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_scalar.default,
    ]
    gm = exported_program.graph_module
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in assert_functions:
            assert len(n.users) == 0
            print("Removing ", n.name)
            gm.graph.erase_node(n)
    gm.recompile()


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_length: int,
        kv_update_method: int,
        use_static_select_in_mask: bool,
    ):
        super().__init__()

        self.kv_update_method = kv_update_method
        self.use_static_select_in_mask = use_static_select_in_mask

        self.use_dynamic_shapes = self.kv_update_method in [1, 4]
        if self.use_dynamic_shapes:
            assert not self.use_static_select_in_mask

        self.kv_io = False
        if self.kv_update_method == 4:
            self.kv_io = True

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        assert n_heads % n_kv_heads == 0
        self.n_rep = n_heads // n_kv_heads

        self.max_seq_length = max_seq_length

        self.max_batch_size = 1

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        cache_shape = (
            self.max_batch_size,
            self.n_heads,
            self.max_seq_length,
            self.head_dim,
        )
        if not self.kv_io:
            self.register_buffer(
                "k_cache",
                torch.zeros(cache_shape, dtype=torch.float32, device="cpu"),
                persistent=False,
            )
            self.register_buffer(
                "v_cache", torch.zeros(cache_shape, dtype=torch.float32, device="cpu")
            )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool),
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,  # (bsz, seqlen, dim)
        input_pos: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        assert bsz <= self.max_batch_size
        assert dim == self.dim

        if self.use_static_select_in_mask:
            attn_mask = self.mask[input_pos.reshape(-1), :]
            assert attn_mask.dim() == 2
        else:
            input_pos_item = input_pos[-1].item()
            torch._check_is_size(input_pos_item)
            torch._check(input_pos_item + seqlen <= self.max_seq_length)
            attn_mask = self.mask.narrow(0, input_pos_item, seqlen)
            assert attn_mask.dim() == 2

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # SDPA expects (bsz, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update cache
        # k, v = self.update_kv_cache(input_pos_item, k, v, k_cache, v_cache)
        k, v = self.update_kv_cache(input_pos, k, v, k_cache, v_cache)

        # Expand KV to match Q for SDPA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # SDPA
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        # y = torch.ops.coreml.sdpa(q, k, v, attn_mask)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y

    def seqlen(self):
        seqlen = 2
        if self.kv_update_method in [2, 3]:
            seqlen = 1

        if self.kv_update_method in [5, 6]:
            seqlen = 10

        return seqlen

    def args(self):
        seqlen = self.seqlen()

        ret = [
            torch.ones(self.max_batch_size, seqlen, self.dim, dtype=torch.float32),
        ]
        if self.kv_update_method in [6]:
            ret.append(
                torch.tensor(
                    [i for i in range(self.seqlen())], dtype=torch.int64
                ).reshape(-1)
            )
        else:
            ret.append(torch.tensor([0], dtype=torch.int64).reshape(1, -1))

        if self.kv_io:
            ret = ret + [
                torch.zeros(
                    self.max_batch_size,
                    self.n_heads,
                    self.max_seq_length,
                    self.head_dim,
                    dtype=torch.float32,
                ),
                torch.zeros(
                    self.max_batch_size,
                    self.n_heads,
                    self.max_seq_length,
                    self.head_dim,
                    dtype=torch.float32,
                ),
            ]

        return tuple(ret)

    def ct_args(self, seqlens, default):
        assert default in seqlens
        ret = [ct.TensorType(shape=t.shape) for t in self.args()]

        if len(seqlens) > 1:
            ret[0] = ct.TensorType(
                shape=ct.EnumeratedShapes(
                    shapes=[(1, s, self.dim) for s in seqlens],
                    default=(1, default, self.dim),
                )
            )
        else:
            ret[0] = ct.TensorType(shape=(1, seqlens[0], self.dim))
        return ret

    def dynamic_shapes(self):
        seqlen = torch.export.Dim(name="seqlen", min=1, max=self.max_seq_length)
        ret = [{1: seqlen}]
        ret = ret + [{} for _ in range(len(self.args()) - len(ret))]
        return ret

    def export_kwargs(self):
        ret = {
            "args": self.args(),
        }
        if self.use_dynamic_shapes:
            ret["dynamic_shapes"] = self.dynamic_shapes()

        return ret

    def update_kv_cache(self, input_pos, k_val, v_val, k_cache, v_cache):
        if not self.kv_io:
            assert k_cache is None
            assert v_cache is None

        if self.kv_update_method == 1:
            return self.update_kv_cache1(input_pos, k_val, v_val)
        elif self.kv_update_method == 2:
            return self.update_kv_cache2(input_pos, k_val, v_val)
        elif self.kv_update_method == 3:
            return self.update_kv_cache3(input_pos, k_val, v_val)
        elif self.kv_update_method == 4:
            return self.update_kv_cache4(input_pos, k_val, v_val, k_cache, v_cache)
        elif self.kv_update_method == 5:
            return self.update_kv_cache5(input_pos, k_val, v_val)
        elif self.kv_update_method == 6:
            return self.update_kv_cache6(input_pos, k_val, v_val)

        assert False

    def update_kv_cache1(self, input_pos, k_val, v_val):
        assert not self.kv_io
        input_pos_item = input_pos[0].item()
        seq_length = k_val.size(2)

        torch._check_is_size(input_pos_item)
        torch._check(input_pos_item >= 0)
        torch._check(input_pos_item + seq_length <= self.max_seq_length)

        narrowed_k = self.k_cache.narrow(2, input_pos_item, seq_length)
        narrowed_k.copy_(k_val)

        narrowed_v = self.v_cache.narrow(2, input_pos_item, seq_length)
        narrowed_v.copy_(v_val)

        return self.k_cache, self.v_cache

    def update_kv_cache2(self, input_pos, k_val, v_val):
        assert not self.kv_io
        assert input_pos.numel() == 1
        input_pos = input_pos.reshape(-1)
        self.k_cache[:, :, input_pos, :] = k_val
        self.v_cache[:, :, input_pos, :] = v_val
        return self.k_cache, self.v_cache

    def update_kv_cache3(self, input_pos, k_val, v_val):
        assert not self.kv_io
        assert input_pos.numel() == 1
        input_pos = input_pos.reshape(-1)
        torch.ops.aten.index_put_(self.k_cache, [None, None, input_pos], k_val)
        torch.ops.aten.index_put_(self.v_cache, [None, None, input_pos], v_val)
        return self.k_cache, self.v_cache

    def update_kv_cache4(self, input_pos, k_val, v_val, k_cache, v_cache):
        assert k_cache is not None
        assert v_cache is not None
        input_pos_item = input_pos[0].item()
        seq_length = k_val.size(2)

        torch._check_is_size(input_pos_item)
        torch._check(input_pos_item >= 0)
        torch._check(input_pos_item + seq_length <= self.max_seq_length)

        after_length = torch.tensor(
            self.max_seq_length - input_pos_item - seq_length
        ).item()

        k_before = k_cache.narrow(2, 0, input_pos_item)

        k_after = k_cache.narrow(2, input_pos_item + seq_length, after_length)
        k_cache_ret = torch.cat([k_before, k_val, k_after], dim=2)
        torch._check(k_cache_ret.size(2) == self.max_seq_length)

        v_before = v_cache.narrow(2, 0, input_pos_item)
        v_after = v_cache.narrow(2, input_pos_item + seq_length, after_length)
        v_cache_ret = torch.cat([v_before, v_val, v_after], dim=2)
        torch._check(v_cache_ret.size(2) == self.max_seq_length)

        return k_cache_ret, v_cache_ret

    def update_kv_cache5(self, input_pos, k_val, v_val):
        assert not self.kv_io
        assert input_pos.numel() == 1
        input_pos = input_pos.reshape(-1)
        self.k_cache[:, :, input_pos : (input_pos + self.seqlen()), :] = k_val
        self.v_cache[:, :, input_pos : (input_pos + self.seqlen()), :] = v_val
        return self.k_cache, self.v_cache

    def update_kv_cache6(self, input_pos, k_val, v_val):
        assert not self.kv_io
        assert input_pos.numel() == self.seqlen()
        self.k_cache[:, :, input_pos, :] = k_val
        self.v_cache[:, :, input_pos, :] = v_val
        return self.k_cache, self.v_cache


########################################################################################################################
# Export attention model for CoreML
########################################################################################################################

with torch.no_grad():
    attention = Attention(
        dim=4096,
        n_heads=32,
        n_kv_heads=32,
        max_seq_length=512,
        # Change kv_update_method to 1, 2, 3, or 4 to test different update methods
        kv_update_method=4,
        use_static_select_in_mask=False,
    )
    args = attention.args()
    attention(*args)
    exported_program = torch.export.export(attention, **attention.export_kwargs())

print(exported_program)
remove_graph_asserts(exported_program)
mlprog = ct.convert(
    exported_program,
    # Uncomment to enable enumerated shapes in CoreML
    # inputs=attention.ct_args(seqlens=[1, 128], default=128),
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    compute_precision=ct.precision.FLOAT16,
    # compute_precision=ct.precision.FLOAT32,
)


# import coremltools.optimize as cto

# op_config = cto.coreml.OpLinearQuantizerConfig(
#     mode="linear_symmetric",
#     dtype="int4",
#     granularity="per_channel",
# )
# config = cto.coreml.OptimizationConfig(global_config=op_config)
# mlprog_compressed = cto.coreml.linear_quantize_weights(mlprog, config=config)
