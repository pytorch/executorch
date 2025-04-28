import torch
import torch.nn as nn
from models import model_dict
import torch_mlir.fx
import os
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export.experimental import _export_forward_backward
from torch._decomp import get_decompositions, register_decomposition
from torch_mlir.extras.fx_decomp_util import DEFAULT_DECOMPOSITIONS
from torch.nn.attention import sdpa_kernel, SDPBackend
import copy
import subprocess
import time
from glob import glob
from mlir_inject_resources import patch_file


ph_input = torch.randn(1, 3, 224, 224)
ph_label = torch.randint(0, 10, (1,))

class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)

@register_decomposition(torch.ops.prims.broadcast_in_dim.default)
def _bcast_dim_decomp(x, shape, b_dims):
    for i, s in enumerate(shape):
        if i not in b_dims:
            x = x.unsqueeze(i)
    return x.expand(shape).clone()

@register_decomposition(torch.ops.prims.div.default)
def _prims_div(a, b, rounding_mode=None):
    return torch.div(a, b, rounding_mode=rounding_mode)

# 2. prims.sum -> aten.sum
@register_decomposition(torch.ops.prims.sum.default)
def _prims_sum(x, dims, dtype=None):
    return torch.sum(x, dim=dims, dtype=dtype)

# 3. prims.var -> aten.var
@register_decomposition(torch.ops.prims.var.default)
def _prims_var(x, dims, correction=0, keepdim=False):
    unbiased = correction != 0
    return torch.var(x, dim=dims, unbiased=unbiased, keepdim=keepdim)

# Remove bernoulli RNG
@register_decomposition(torch.ops.aten.bernoulli.p)
def _bernoulli_p(x, p, generator=None):
    scale = 1.0 / (1.0 - p)
    return torch.full_like(x, scale).to(x.dtype)

def cannonicalize_empty_permute(ep):
    for node in ep.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.empty_permuted.out
        ):
            print("found empty_permute: ", node)
            empty_permuted_node = node
            with ep.graph.inserting_before(empty_permuted_node):
                empty_node = ep.graph.create_node(
                    "call_function",
                    torch.ops.aten.empty.memory_format,
                    (node.args[0],),
                    empty_permuted_node.kwargs,
                )
                permute_node = ep.graph.create_node(
                    "call_function",
                    torch.ops.aten.permute,
                    (empty_node, node.args[1]),
                )
                for user in empty_permuted_node.users.copy():
                    user.replace_input_with(empty_permuted_node, permute_node)
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.empty_permuted.default
        ):
            print("found empty_permute default: ", node)
            empty_permuted_node = node
            with ep.graph.inserting_before(empty_permuted_node):
                empty_node = ep.graph.create_node(
                    "call_function",
                    torch.ops.aten.empty.memory_format,
                    (node.args[0],),
                    empty_permuted_node.kwargs,
                )
                permute_node = ep.graph.create_node(
                    "call_function",
                    torch.ops.aten.permute.default,
                    (empty_node, node.args[1]),
                )
                for user in empty_permuted_node.users.copy():
                    user.replace_input_with(empty_permuted_node, permute_node)
    return ep


def compile_executorch(ep, prefix):
    xnn_et = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(f"{prefix}.pte", "wb") as f:
        f.write(xnn_et.buffer)

def safe_cmd(cmd, timeout_sec=150, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )

            stderr = result.stderr.lower()
            stdout = result.stdout.lower()

            if result.returncode == 0:
                print("✅ Compilation succeeded.")
                return  # 정상 종료

            if "double free" in stderr or "double free" in stdout:
                print("⚠️ Detected 'double free' error. Retrying...")
                time.sleep(1)
                continue

            # 다른 오류면 바로 종료
            print("❌ Compilation failed:")
            print(stderr)
            break

        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout after {timeout_sec} seconds. Retrying...")
            continue

    print("❌ Compilation failed after max retries.")
    return

def compile_iree(ep, prefix, decomposition_table=None):
    print("Exporting MLIR")
    mlir = torch_mlir.fx.export_and_import(ep, output_type="raw", decomposition_table=decomposition_table)
    print(mlir, file=open(f"{prefix}.mlir", "w"))
    cmd = ["/root/iree-latest-build/install/bin/iree-compile",
        "--iree-input-type=torch",
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
        "--mlir-elide-elementsattrs-if-larger=1",
        "--mlir-elide-resource-strings-if-larger=1",
        f"--iree-hal-dump-executable-benchmarks-to={prefix}_dispatch",
        f"--dump-compilation-phases-to={prefix}_phases",
        f"{prefix}.mlir",
        "-o",
        f"{prefix}.vmfb"
    ]
    print("\n\t".join(cmd))
    safe_cmd(cmd)
    for dispatch in glob(f"{prefix}_dispatch/*.mlir"):
        patch_file(dispatch, dispatch)

def export_inference(model, prefix, et=True, iree=True):
    ep = torch.export.export(model, (ph_input, ))
    print(ep.graph_module.graph, file=open(f"{prefix}_graph.txt", "w"))
    if et:
        compile_executorch(ep, prefix)
    if iree:
        decomposition_table = get_decompositions([
            *DEFAULT_DECOMPOSITIONS,
            torch.ops.aten.as_strided
        ])
        compile_iree(ep, prefix, decomposition_table)


def export_training(model, prefix, et=True, iree=True):
    model = TrainingNet(model)
    if et: # faster than deepcopy
        ep = torch.export.export(model, (ph_input, ph_label), strict=False)
        ep = _export_forward_backward(ep)
        ep = cannonicalize_empty_permute(ep)
        print(ep.graph_module.graph, file=open(f"{prefix}_graph.txt", "w"))
        print("Training Executorch - EP mutate done")
        decomposition_table = get_decompositions([
            torch.ops.aten._native_batch_norm_legit_functional.default,
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten.var_mean.correction,
            torch.ops.prims.broadcast_in_dim.default,
            torch.ops.prims.div.default,      # ★
            torch.ops.prims.sum.default,      # ★
            torch.ops.prims.var.default,
            torch.ops.aten.bernoulli.p,
            torch.ops.aten.empty_strided.out,
            torch.ops.aten.convolution_backward
        ])
        ep = ep.run_decompositions(decomposition_table)
        compile_executorch(ep, prefix)
        print("Training Executorch - compilation done")
    if iree:
        ep = torch.export.export(model, (ph_input, ph_label), strict=False)
        ep = _export_forward_backward(ep)
        ep = cannonicalize_empty_permute(ep)
        print("Training IREE - EP mutate done")
        decomposition_table = get_decompositions([
            *DEFAULT_DECOMPOSITIONS,
            torch.ops.aten.convolution_backward,
            torch.ops.aten.bernoulli.p,
            torch.ops.aten.any.dim,
            torch.ops.aten.as_strided
        ])
        compile_iree(ep, prefix, decomposition_table)
        print("Training IREE - compilation done")

def export_model(model_name, train=False, et=True, iree=True):
    print(f"Exporting {model_name} {'train' if train else 'inf'}")
    model = model_dict[model_name](train)
    prefix = f"./{model_name}/{'train' if train else 'inf'}"
    if not os.path.exists(f"./{model_name}"):
        os.makedirs(f"./{model_name}")
    if not train:
        export_inference(model, prefix, et, iree)
    else:
        export_training(model, prefix, et, iree)

if __name__ == "__main__":
    for train in [False, True]:
        # export_model("vgg", train)
        # export_model("alexnet", train)
        export_model("mobilenet_v2", False, et=False)
        # export_model("efficientnet_b0", train)
        # export_model("resnet18", train)
        with sdpa_kernel([SDPBackend.MATH]):
            # export_model("vit_b_16", train)
            # export_model("vit_tiny_patch16_224", train, et=False)
            # export_model("mobilevit_s", train, et=False)
            pass