# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib
import tempfile

import torch


class AmdPointwiseModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * torch.sigmoid(x + 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a small Triton-backed AOTI delegate for an AMD GPU."
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args()

    if torch.version.hip is None:
        raise RuntimeError(
            "export_amd_pointwise.py requires a ROCm-enabled PyTorch build"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AOTInductor ROCm lowering requires a visible AMD GPU to select and "
            "compile the initial Triton kernel; C++ execution is not required"
        )

    # Match AOTInductor's active device.
    device_index = torch.cuda.current_device()
    device = torch.cuda.get_device_properties(device_index)
    arch = device.gcnArchName.split(":", 1)[0]
    # Use the targets supported by the installed PyTorch toolchain.
    supported = torch.cuda.get_arch_list()
    if arch not in supported:
        raise ValueError(
            f"{arch} is not in this PyTorch build's supported targets: "
            f"{', '.join(supported)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(
        tempfile.mkdtemp(prefix="torchinductor_amd_", dir=args.output_dir)
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.backend.compile_spec_schema import CompileSpec

    module = AmdPointwiseModule().eval()
    example_inputs = (torch.randn(1024, 1024),)
    exported_program = torch.export.export(module, example_inputs, strict=True)

    compile_specs = [
        CudaBackend.generate_method_name_compile_spec("forward"),
        CompileSpec("target_device", f"cuda:{device_index}".encode()),
        CompileSpec("triton_kernel_mode", b"ON"),
        CompileSpec("max_autotune", b"OFF"),
        CompileSpec("autotune_at_compile_time", b"OFF"),
    ]
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[CudaPartitioner(compile_specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )

    delegated = any(
        node.op == "call_function" and "executorch_call_delegate" in str(node.target)
        for node in edge_program.exported_program().graph.nodes
    )
    if not delegated:
        raise RuntimeError(
            "the exported program does not contain an AOTI delegate call"
        )

    triton_sources = []
    for source_path in cache_dir.rglob("*.py"):
        source = source_path.read_text(errors="ignore")
        if "async_compile.triton(" in source or "@triton_heuristics." in source:
            triton_sources.append(source_path)
    if not triton_sources:
        raise RuntimeError(
            f"Inductor did not generate a Triton kernel under {cache_dir}"
        )

    executorch_program = edge_program.to_executorch()
    pte_path = args.output_dir / "amd_triton.pte"
    with pte_path.open("wb") as output_file:
        executorch_program.write_to_file(output_file)
    executorch_program.write_tensor_data_to_file(args.output_dir)

    ptd_path = args.output_dir / "aoti_cuda_blob.ptd"
    if not ptd_path.is_file():
        raise RuntimeError(f"expected AOTI data file was not created: {ptd_path}")

    # Verify the embedded code object matches the active device.
    triple = f"amdgcn-amd-amdhsa--{arch}".encode()
    if triple not in pte_path.read_bytes():
        raise RuntimeError(f"{pte_path} embeds no code object for {arch}")

    print(f"ROCm: {torch.version.hip}")
    print(f"Compile GPU: cuda:{device_index} {device.name} ({device.gcnArchName})")
    print(f"Compile architecture: {arch}")
    print(f"ExecuTorch program: {pte_path}")
    print(f"AOTI data: {ptd_path}")
    print("Triton sources:")
    for source_path in triton_sources:
        print(f"  {source_path}")


if __name__ == "__main__":
    main()
