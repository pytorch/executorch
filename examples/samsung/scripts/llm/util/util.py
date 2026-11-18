# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import random
import typing

import numpy as np
import torch
from torch.fx import Interpreter, Node

from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4VisionAttention,
    )
except ImportError:
    # transformers version without gemma4 support
    Gemma4RMSNorm = None
    Gemma4VisionAttention = None

from transformers.models.llama.modeling_llama import LlamaRMSNorm


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_rms_dim(
    name: str,
    child: Gemma4RMSNorm,
    parent: torch.nn.Module,
) -> int:
    """Get the normalization dimension for Gemma4RMSNorm."""
    if hasattr(child, "weight"):
        return child.weight.shape[-1]
    elif "embedding_pre_projection_norm" in name:
        dim = parent.multimodal_hidden_size
    elif "v_norm" in name and (
        Gemma4VisionAttention is not None and isinstance(parent, Gemma4VisionAttention)
    ):
        dim = parent.head_dim
    else:
        assert 1 == 2, f"Unsupport norm layer {name}"
    return dim


# ----------------------------------------------------------------------
# Replace
#     Modified from executorch.examples.models.llama.source_transformation.rms_norm.py
# ----------------------------------------------------------------------
def replace_rms_norm_with_native_rms_norm(
    module: torch.nn.Module,
):
    """Recursively replace Gemma4RMSNorm with torch.nn.RMSNorm."""
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
        elif Gemma4RMSNorm is not None and isinstance(child, Gemma4RMSNorm):
            param = next(module.parameters())
            dim = _get_rms_dim(name, child, module)
            rms_norm = torch.nn.RMSNorm(
                dim, eps=child.eps, elementwise_affine=child.with_scale
            ).to(param)
            if child.with_scale:
                rms_norm.weight = torch.nn.Parameter(child.weight.data)
            else:
                rms_norm.weight = torch.nn.Parameter(torch.ones(dim))
            rms_norm = rms_norm.to(param)
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


def adding_dequant(gm, state_dict):
    dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default

    def convert_layer_name(name, exceptions):
        # Sort by length descending to avoid partial matches
        exceptions_sorted = sorted(exceptions, key=len, reverse=True)

        # Step 1: protect exceptions with placeholder
        protected = name
        placeholders = {}
        for i, exc in enumerate(exceptions_sorted):
            ph = f"\x00{i}\x00"
            placeholders[ph] = exc
            protected = protected.replace(exc, ph)

        # Step 2: replace remaining '_' with '.'
        replaced = protected.replace("_", ".")

        # Step 3: restore exceptions
        for ph, exc in placeholders.items():
            replaced = replaced.replace(ph, exc)

        return replaced

    EXCEPTIONS = [
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "embed_tokens",
        "lm_head",
    ]

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.linear.default:
            wei_obs = node.args[1]
            name = wei_obs.args[0]  # layers_0_self_attn_q_proj_weight
            if str(name) == "lm_head_weight":
                break
            name_dict = convert_layer_name(str(name), EXCEPTIONS)
            weight = state_dict["model." + name_dict]
            weight_scale = state_dict["model." + name_dict + "_scale"]  # .squeeze()
            weight_zp = torch.zeros_like(weight_scale)

            frozen_name = str(name) + "_frozen_param_"
            scale_name = str(name) + "_scale_"
            zp_name = str(name) + "_zero_point_"
            gm.register_buffer(
                frozen_name, weight.to(torch.int8)
            )  # wei_observer.w_int.to(torch.int8))
            gm.register_buffer(scale_name, weight_scale)  # wei_observer.scale)
            gm.register_buffer(
                zp_name, weight_zp
            )  # wei_observer.offset.to(torch.int8))

            graph = gm.graph
            with graph.inserting_after(wei_obs):
                frozen_node = graph.create_node("get_attr", frozen_name, (), {})
                scale_node = graph.create_node("get_attr", scale_name, (), {})
                zp_node = graph.create_node("get_attr", zp_name, (), {})
            with graph.inserting_after(frozen_node):
                dq_node = graph.call_function(
                    dequantize_op,
                    args=(
                        frozen_node,
                        scale_node,
                        zp_node,
                        0,  # axis
                        -8,  # qmin
                        7,  # qmax
                        torch.int8,  # dtype
                    ),
                )
            wei_obs.replace_all_uses_with(dq_node)
            gm.graph.erase_node(wei_obs)

    gm.graph.lint()
    gm.recompile()
    return gm


class IntermediateTracer(Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.node_inout = {"input": {}, "output": {}, "connect": {}}

    def clone_tensor(self, obj):
        import types

        if isinstance(obj, torch.Tensor):
            return self._materialize_tensor(obj)
        elif isinstance(obj, tuple):
            return tuple(self.clone_tensor(x) for x in obj)
        elif isinstance(obj, list):
            return [self.clone_tensor(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.clone_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, types.ModuleType):
            return f"<Module: {obj.__name__}>"
        return None

    def _materialize_tensor(self, t):
        # Turn any torch.Tensor (including non-picklable subclasses such as
        # BatchedTensorImpl / functional tensors that expose no real storage)
        # into a plain CPU tensor. `torch.save` stores tensors via their
        # storage, so a storage-less subclass raises NotImplementedError at
        # pickling time. We guard against that and fall back to picklable
        # metadata so the dump never crashes.
        for fn in (
            lambda: t.detach().cpu().clone(),
            lambda: t.cpu(),
            lambda: t.to("cpu"),
            lambda: t.detach(),
        ):
            try:
                out = fn()
                out.untyped_storage()  # rejects storage-less tensor subclasses
                return out.cpu().clone()
            except Exception:
                continue
        try:
            return {
                "_undumpable_tensor": True,
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "device": str(t.device),
            }
        except Exception:
            return None

    def run_node(self, n: Node):
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        self.node_inout["connect"][n.name] = {
            "ifm": [str(name) for name in n.args],
            "ofm": [str(name) for name in n.users],
        }

        self.node_inout["input"][n.name] = (
            self.clone_tensor(args),
            self.clone_tensor(kwargs),
        )

        result = super().run_node(n)
        self.node_inout["output"][n.name] = self.clone_tensor(result)
        return result

    def save_inout(self, path):
        torch.save(self.node_inout, path)

    def get_flat_inputs_for_interpreter(self, edge_program, user_inputs):
        flat_inputs = []

        sig = edge_program.graph_signature
        state_dict = edge_program.state_dict

        if isinstance(user_inputs, torch.Tensor):
            user_inputs = [user_inputs]
        else:
            user_inputs = list(user_inputs)

        user_input_idx = 0

        for input_spec in sig.input_specs:
            kind = input_spec.kind.name
            target = input_spec.target

            if kind == "PARAMETER" or kind == "BUFFER":
                if target in state_dict:
                    flat_inputs.append(state_dict[target])
                else:
                    raise KeyError(f"Can't get the parameter in state_dict: {target}")

            elif kind == "USER_INPUT":
                if user_input_idx < len(user_inputs):
                    flat_inputs.append(user_inputs[user_input_idx])
                    user_input_idx += 1
                else:
                    raise RuntimeError("The number of inputs is wrong")

            else:
                if (
                    hasattr(edge_program, "constants")
                    and target in edge_program.constants
                ):
                    flat_inputs.append(edge_program.constants[target])
                else:
                    pass

        return flat_inputs
