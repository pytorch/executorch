# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch

from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


# ----------------------------------------------------------------------
# Replace
#     Modified from executorch.examples.models.llama.source_transformation.rms_norm.py
# ----------------------------------------------------------------------
def replace_rms_norm_with_native_rms_norm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, LlamaRMSNorm):
            rms_norm = torch.nn.RMSNorm(
                child.weight.shape[-1], eps=child.variance_epsilon
            )
            rms_norm.weight = child.weight
            setattr(module, name, rms_norm)
        elif isinstance(child, Gemma3RMSNorm):
            rms_norm = torch.nn.RMSNorm(child.weight.shape[-1], eps=child.eps)
            rms_norm.weight = torch.nn.Parameter(child.weight + 1.0)
            setattr(module, name, rms_norm)
        else:
            replace_rms_norm_with_native_rms_norm(child)
    return module


def _fix_device(v):
    """Map any CUDA device (torch.device or str) to CPU, else return unchanged."""
    if isinstance(v, torch.device):
        return torch.device("cpu") if v.type == "cuda" else v
    if isinstance(v, str) and (v == "cuda" or v.startswith("cuda:")):
        return "cpu"
    return v


def _fix_container(c):
    """Recursively rewrite CUDA devices in nested tuple/list/dict containers."""
    if isinstance(c, (torch.device, str)):
        return _fix_device(c)
    if isinstance(c, tuple):
        out = tuple(_fix_container(x) for x in c)
        return out if any(n is not x for x, n in zip(c, out)) else c
    if isinstance(c, list):
        out = [_fix_container(x) for x in c]
        return out if any(n is not x for x, n in zip(c, out)) else c
    if isinstance(c, dict):
        out = {k: _fix_container(v) for k, v in c.items()}
        return out if any(n is not v for v, n in zip(c.values(), out.values())) else c
    return c


def removing_gpu_node_in_graph(model):
    """Make an exported/quantized GraphModule run (and re-export) on CPU.

    The vision encoder is traced on GPU, so its graph bakes in CUDA: e.g. the
    rotary-emb `_to_copy(device=cuda)` and device-creating ops, and the
    non-persistent `inv_freq` buffer lifted as a graph constant. Plain
    `model.to('cpu')` cannot fix these. This rewrites every baked CUDA device
    op/arg to CPU and moves every tensor constant/buffer to CPU, recursively
    through nested submodules (e.g. higher-order-op submodules).
    """
    # Rewrite device args/kwargs in every nested GraphModule graph.
    for gm in list(model.modules()):
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        graph = gm.graph
        for node in list(graph.nodes):
            if node.target == torch.ops.aten._assert_tensor_metadata.default:
                node.replace_all_uses_with(node.args[0])  # bypass
                graph.erase_node(node)
                continue
            new_args = _fix_container(node.args)
            if new_args is not node.args:
                node.args = new_args
            new_kwargs = _fix_container(node.kwargs)
            if new_kwargs is not node.kwargs:
                node.kwargs = new_kwargs
        graph.eliminate_dead_code()
        gm.recompile()

    # Move every tensor constant/buffer (including non-persistent ones that
    # `.to()` normally skips) to CPU across all submodules.
    for sub in list(model.modules()):
        for name, val in list(sub.__dict__.items()):
            if isinstance(val, torch.Tensor) and val.is_cuda:
                sub.__dict__[name] = val.cpu()
        for name, buf in list(sub.named_buffers(recurse=False)):
            if buf.is_cuda:
                sub._buffers[name] = buf.cpu()

    return model


def build_calib_dataset(tokenizer, sample_number=30, max_length=1024):
    from datasets import load_dataset

    texts = []
    chunk_size = max_length - 10
    if chunk_size <= 0:
        raise ValueError(f"max_length ({max_length}) must be greater than 10.")
    dataset = load_dataset(  # nosec B615
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )
    full_text = "\n\n".join([x["text"] for x in dataset if x["text"].strip()])
    encodings = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids[0]
    bos_token_id = tokenizer.bos_token_id
    if sample_number * chunk_size > len(input_ids):
        raise ValueError(
            f"{sample_number} calibration samples of {chunk_size} tokens exceed "
            f"the {len(input_ids)} tokens of wikitext-2."
        )
    for idx in range(sample_number):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size
        chunk_tokens = input_ids[start_idx:end_idx]
        bos_tensor = torch.tensor([bos_token_id]).to(chunk_tokens)
        final_chunk = torch.cat([bos_tensor, chunk_tokens]).unsqueeze(0)
        texts.append({"input_ids": final_chunk})
    return texts


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.to(dtype=torch.float32)
        # pyre-fixme[58]: `*` is not supported for operand types `Tensor` and
        #  `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`.
        linear.weight.data = (W_ * layernorm.weight.to(dtype=torch.float32)).to(
            linear_dtype
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float32)
                )
            linear.bias.data = linear.bias.data.to(dtype=torch.float32) + torch.matmul(
                W_,
                # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                layernorm.bias.to(dtype=torch.float32),
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model: torch.nn.Module):
    # Embedding fusion
    for W in [model.embed_tokens]:
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `weight`.
        W_ = W.weight.data.to(dtype=torch.float32)
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `weight`.
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    # pyre-fixme[29]:
    #  `Union[BoundMethod[typing.Callable(torch._tensor.Tensor.__iter__)[[Named(self,
    #  torch._tensor.Tensor)], typing.Any], torch._tensor.Tensor],
    #  torch._tensor.Tensor, torch.nn.modules.module.Module]` is not a function.
    for layer in model.layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )

        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            W_norm, dtype=torch.float32
        )
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm, dtype=torch.float32)

    fuse_ln_linear(
        # pyre-fixme[6]: For 1st argument expected `Module` but got `Union[Tensor,
        #  Module]`.
        model.norm,
        # pyre-fixme[6]: For 2nd argument expected `Iterable[Linear]` but got
        #  `Iterable[Union[Tensor, Module]]`.
        [model.lm_head],
    )
    # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
    #  `weight`.
    W_norm = model.norm.weight.data
    model.norm.weight.data = torch.ones_like(W_norm, dtype=torch.float32)

    return model


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def rotate_self_attn_inputs(layer, R1) -> None:
    # Rotate the q_proj, k_proj and v_proj matrices of the self-self_attn layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def rotate_self_attn_output(layer, R1) -> None:
    # Rotate output matrix of the self-self_attn layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(R1.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float32)
        W.bias.data = torch.matmul(R1.T, b).to(dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(R1.T, W_).to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float32)
        W.bias.data = torch.matmul(R1.T, b).to(dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def rotate_ov_proj(layer, head_dim, R2=None):
    W = layer.self_attn.v_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32).t()
    transposed_shape = W_.shape
    temp = W_.reshape(-1, transposed_shape[-1] // head_dim, head_dim)
    temp = temp.to(torch.float32) @ R2
    W_ = temp.reshape(transposed_shape).t()
    W.weight.data = W_.to(dtype=dtype)

    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float32)
    init_shape = W_.shape
    temp = W_.reshape(-1, init_shape[-1] // head_dim, head_dim)
    temp = temp.to(torch.float32) @ R2
    W_ = temp.reshape(init_shape)
    W.weight.data = W_.to(dtype=dtype)


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    import gc

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()


def get_model_with_r1_r2(optimized_rotation_path: str):
    return lambda model: apply_spin_quant_r1_r2(model, optimized_rotation_path)


def apply_spin_quant_r1_r2(model: torch.nn.Module, optimized_rotation_path: str):
    device = next(model.parameters()).device
    optimized_rotation = torch.load(optimized_rotation_path, weights_only=True)
    R1 = optimized_rotation["R1"].to(torch.float32).to(device)
    config = model.config
    # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
    #  `n_heads`.
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    cleanup_memory()

    # pyre-fixme[6]: For 1st argument expected `Iterable[Variable[_T]]` but got
    #  `Union[Tensor, Module]`.
    for idx, layer in enumerate(model.layers):
        key = f"model.layers.{idx}.self_attn.R2"
        R2 = optimized_rotation[key].to(torch.float32).to(device)
        rotate_self_attn_inputs(layer, R1)
        rotate_self_attn_output(layer, R1)
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)
        rotate_ov_proj(layer, head_dim, R2=R2)
    return model
