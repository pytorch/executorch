# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import inspect
import os
import sys

from typing import Dict, final, Optional, Sequence, Type

import executorch.exir as exir

import torch
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import (
    BackendDetails,
    DelegateScratchSpec,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (  # noqa: F401
    ExecutorBackend,
)
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
from executorch.exir.program import ExecutorchProgramManager
from torch import nn
from torch.export import export

"""Traces and exports delegated nn.Modules to ExecuTorch .pte program files.

Creates two versions of each file:
- <module-name>.pte: Delegate data stored in segments outside of the flatbuffer data.
- <module-name>-nosegments.pte: Delegate data is stored directly in the flatbuffer data.

This tool mainly exists to export programs for C++ tests, but can also
be used to export models manually.
"""

#
# Modules
#


class ModuleAddMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, a: torch.Tensor, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        y: torch.Tensor = torch.mm(a, x)
        z: torch.Tensor = torch.add(y, b)
        return z

    def get_random_inputs(self) -> Sequence[torch.Tensor]:
        return (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))


class ModuleAddLarge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x: torch.Tensor = torch.add(a, b)
        y: torch.Tensor = torch.add(x, c)
        z: torch.Tensor = torch.add(x, y)
        return z

    def get_random_inputs(self) -> Sequence[torch.Tensor]:
        n = 10  # to create a large tensor
        return (torch.ones(n, n, n), 2 * torch.ones(n, n, n), 3 * torch.ones(n, n, n))


class ModuleSubLarge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x: torch.Tensor = torch.sub(a, b)
        y: torch.Tensor = torch.sub(x, c)
        z: torch.Tensor = torch.sub(x, y)
        w: torch.Tensor = torch.sub(z, c)
        return w

    def get_random_inputs(self) -> Sequence[torch.Tensor]:
        n = 10  # to create a large tensor
        return (torch.ones(n, n, n), 2 * torch.ones(n, n, n), 3 * torch.ones(n, n, n))


class ModuleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(3),)


#
# XNNPACK weights cache modules
#
# The weights cache keys packed kernels by the name of their unpacked bytes,
# and the exporter names constants by content hash. A weight and its double
# quantize to the same int8 codes under symmetric quantization, so they share
# one name with different scales, and one weight feeding two nodes with
# different activation quantization needs two packings under one name. Each
# module below pairs its kernels that way so a run with the cache on can be
# compared against one with it off.
#
# Statically quantized kernels are packed with their requantization scale,
# input scale * kernel scale / output scale, which a doubled kernel or a scaled
# input leaves unchanged once the output is calibrated on its own. The static
# modules therefore concatenate the paired outputs: the quantizer gives the
# inputs of a cat one shared output scale, so the pairs requantize apart.
#


def _export_pt2e(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    is_per_channel: bool,
    is_dynamic: bool = False,
):
    from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
        DuplicateDynamicQuantChainPass,
    )
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=is_per_channel, is_dynamic=is_dynamic
        )
    )
    prepared = prepare_pt2e(export(module, args=inputs).module(), quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared)
    if is_dynamic:
        DuplicateDynamicQuantChainPass()(converted)
    return export(converted, args=inputs)


def _copy_scaled(dst: torch.Tensor, src: torch.Tensor, factor: float) -> None:
    with torch.no_grad():
        dst.copy_(factor * src)


class ModuleQuantSharedLinear(nn.Module):
    """Static per-channel int8: `doubled` shares `linear`'s int8 codes with
    doubled scales, `twin` shares its codes and scales but consumes an input
    with a different scale, and `linear` itself feeds two nodes."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(16, 8, bias=False)
        self.doubled = nn.Linear(16, 8, bias=False)
        self.twin = nn.Linear(16, 8, bias=False)
        _copy_scaled(self.doubled.weight, self.linear.weight, 2.0)
        _copy_scaled(self.twin.weight, self.linear.weight, 1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.cat(
            [self.linear(x), self.doubled(x), self.twin(y), self.linear(y)], dim=1
        )

    def get_random_inputs(self):
        torch.manual_seed(1)
        x = torch.randn(4, 16)
        return (x, 4 * x)

    def export_quantized(self, inputs):
        return _export_pt2e(self, inputs, is_per_channel=True)


class ModuleQuantPerTensorLinear(nn.Module):
    """Static per-tensor int8: `doubled` shares `linear`'s int8 codes with a
    doubled scale."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(16, 8, bias=False)
        self.doubled = nn.Linear(16, 8, bias=False)
        _copy_scaled(self.doubled.weight, self.linear.weight, 2.0)

    def forward(self, x: torch.Tensor):
        return torch.cat([self.linear(x), self.doubled(x)], dim=1)

    def get_random_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 16),)

    def export_quantized(self, inputs):
        return _export_pt2e(self, inputs, is_per_channel=False)


class ModuleQuantDoubledConv(nn.Module):
    """Static per-channel int8 convolution, depthwise convolution and
    transposed convolution, each with a doubled twin. (The quantizer leaves a
    convolution weight's second consumer unquantized, so the shared-weight
    case lives in ModuleQuantSharedLinear.)"""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.conv_doubled = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
        self.dw_doubled = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
        self.deconv = nn.ConvTranspose2d(4, 4, 2, stride=2, bias=False)
        self.deconv_doubled = nn.ConvTranspose2d(4, 4, 2, stride=2, bias=False)
        _copy_scaled(self.conv_doubled.weight, self.conv.weight, 2.0)
        _copy_scaled(self.dw_doubled.weight, self.dw.weight, 2.0)
        _copy_scaled(self.deconv_doubled.weight, self.deconv.weight, 2.0)

    def forward(self, x: torch.Tensor):
        return (
            torch.cat([self.conv(x), self.conv_doubled(x)], dim=1),
            torch.cat([self.dw(x), self.dw_doubled(x)], dim=1),
            torch.cat([self.deconv(x), self.deconv_doubled(x)], dim=1),
        )

    def get_random_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 4, 8, 8),)

    def export_quantized(self, inputs):
        return _export_pt2e(self, inputs, is_per_channel=True)


class ModuleDynQuantSharedLinear(nn.Module):
    """Dynamic per-channel int8: `doubled` shares `linear`'s int8 codes with
    doubled scales, and `linear` feeds two nodes."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(16, 8, bias=False)
        self.doubled = nn.Linear(16, 8, bias=False)
        _copy_scaled(self.doubled.weight, self.linear.weight, 2.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (self.linear(x), self.doubled(x), self.linear(y))

    def get_random_inputs(self):
        torch.manual_seed(1)
        x = torch.randn(4, 16)
        return (x, 4 * x)

    def export_quantized(self, inputs):
        return _export_pt2e(self, inputs, is_per_channel=True, is_dynamic=True)


class ModuleGroupwiseQuantLinear(nn.Module):
    """Dynamic int8 activations with int4 per-channel-group weights: `doubled`
    shares `linear`'s int4 codes with doubled group scales."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(64, 8, bias=False)
        self.doubled = nn.Linear(64, 8, bias=False)
        _copy_scaled(self.doubled.weight, self.linear.weight, 2.0)

    def forward(self, x: torch.Tensor):
        return (self.linear(x), self.doubled(x))

    def get_random_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 64),)

    def export_quantized(self, inputs):
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
            quantize_,
        )
        from torchao.utils import unwrap_tensor_subclass

        quantize_(
            self,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(32)
            ),
        )
        unwrap_tensor_subclass(self)
        return export(self, args=inputs)


#
# Backends
#


@final
class StubBackend(BackendDetails):
    """No-op backend to test serialization/init."""

    # Bytes of memory-planned scratch to declare, as decimal ASCII.
    SCRATCH_BYTES_KEY = "scratch_bytes"

    @staticmethod
    def preprocess(edge_program, compile_specs) -> PreprocessResult:
        scratch_bytes = 0
        for spec in compile_specs:
            if spec.key == StubBackend.SCRATCH_BYTES_KEY:
                scratch_bytes = int(spec.value)
        return PreprocessResult(
            processed_bytes=b"StubBackend:data",
            scratch_specs=(
                [DelegateScratchSpec(nbytes=scratch_bytes)] if scratch_bytes else []
            ),
        )


#
# Program logic
#


def export_module_to_program(
    module_class: Type[nn.Module],
    *,
    backend_id: str,
    extract_delegate_segments: bool,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
    method_name: str = "forward",
    external_constants: bool = False,
    scratch_bytes: int = 0,
) -> ExecutorchProgramManager:
    eager_module = module_class().eval()
    inputs = ()
    if hasattr(eager_module, "get_random_inputs"):
        inputs = eager_module.get_random_inputs()  # type: ignore[operator]

    class WrapperModule(torch.nn.Module):
        def __init__(self, fn, method_name=method_name):
            super().__init__()
            self.fn = fn
            self.method_name = method_name

        def forward(self, *args, **kwargs):
            return getattr(self.fn, self.method_name)(*args, **kwargs)

    if method_name != "forward":
        # Only require wrapper module if we're exporting a specific method other than forward.
        exported_program = export(WrapperModule(eager_module), args=inputs)
    elif hasattr(eager_module, "export_quantized"):
        exported_program = eager_module.export_quantized(inputs)  # type: ignore[operator]
    else:
        exported_program = export(eager_module, args=inputs)

    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    et_config = exir.ExecutorchBackendConfig(
        extract_delegate_segments=extract_delegate_segments,
        constant_tensor_alignment=constant_tensor_alignment,
        delegate_alignment=delegate_alignment,
        external_constants=external_constants,
    )

    if backend_id == "XnnpackBackend":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        if external_constants:
            tagged_module = exported_program.module()
            delegate_external_constants_pass_unlifted(
                module=tagged_module,
                gen_tag_fn=lambda x: module_class.__name__,
            )
            exported_program = export(tagged_module, args=inputs)
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            compile_config=edge_config,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch(config=et_config)
    else:
        edge: exir.EdgeProgramManager = to_edge(exported_program)
        compile_specs = [
            # Just for the demo executor_backend.
            CompileSpec(key="external_constants", value=b"")
        ]
        if scratch_bytes:
            compile_specs.append(
                CompileSpec(
                    key=StubBackend.SCRATCH_BYTES_KEY,
                    value=str(scratch_bytes).encode(),
                )
            )
        lowered_module = to_backend(  # type: ignore[call-arg]
            backend_id,
            edge.exported_program(),
            compile_specs=compile_specs,
        )

        class CompositeModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_module = lowered_module

            def forward(self, *args, **kwargs):
                return self.lowered_module(*args, **kwargs)

        composite_module = CompositeModule()
        composite_module(*inputs)

        executorch_program = to_edge(
            export(composite_module, args=inputs)
        ).to_executorch(config=et_config)

    return executorch_program


def main() -> None:
    known_backend_ids = [
        BackendWithCompilerDemo.__name__,
        StubBackend.__name__,
    ]

    # These args are optimized for genrule usage. There's a lot of startup
    # overhead for this tool, so it's faster to export multiple models at once
    # when possible.
    parser = argparse.ArgumentParser(
        prog="export_delegated_program",
        description="Exports delegated nn.Module models to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--modules",
        help="Comma-separated list of model class names to export; "
        + "e.g., '--modules=ModuleOne,ModuleTwo'",
        type=lambda s: [item.strip() for item in s.split(",")],
    )
    parser.add_argument(
        "--backend_id",
        type=str,
        default=StubBackend.__name__,
        help="ID of the backend to use for delegation; "
        + f"one of {known_backend_ids}",
    )
    parser.add_argument(
        "--inline_delegate_segments",
        action="store_true",
        help="Store delegate data inside the flatbuffer.",
    )
    parser.add_argument(
        "--delegate_alignment", type=int, default=None, help="Delegate alignment."
    )
    parser.add_argument(
        "--external_constants",
        action="store_true",
        help="Export the model with all constants saved to an external file.",
    )
    parser.add_argument(
        "--scratch_bytes",
        type=int,
        default=0,
        help="Bytes of memory-planned scratch for StubBackend to declare.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write <classname>[-<suffix>[...]].pte "
        + "files to.",
    )
    args = parser.parse_args()

    # Find the classes to export. Only looks in this module for now, but could
    # be extended to look in other modules if helpful.
    module_names_to_classes: Dict[str, Type[nn.Module]] = {}
    for module in args.modules:
        module_class = getattr(sys.modules[__name__], module, None)
        if not (inspect.isclass(module_class) and issubclass(module_class, nn.Module)):
            raise NameError(f"Could not find nn.Module class named '{module}'")
        module_names_to_classes[module] = module_class

    # Export and write to the output files.
    os.makedirs(args.outdir, exist_ok=True)
    suffix = ""
    for module_name, module_class in module_names_to_classes.items():
        if args.inline_delegate_segments:
            suffix += "-nosegments"
        if args.delegate_alignment is not None:
            suffix += f"-da{args.delegate_alignment}"
        if args.external_constants:
            suffix += "-e"
        if args.scratch_bytes:
            suffix += "-scratch"
        outfile = os.path.join(args.outdir, f"{module_name}{suffix}.pte")
        executorch_program = export_module_to_program(
            module_class,
            backend_id=args.backend_id,
            extract_delegate_segments=not args.inline_delegate_segments,
            delegate_alignment=args.delegate_alignment,
            external_constants=args.external_constants,
            scratch_bytes=args.scratch_bytes,
        )
        with open(outfile, "wb") as fp:
            fp.write(executorch_program.buffer)
        print(f"Exported {module_name} and wrote program data to {outfile}")
        if args.external_constants:
            print(f"Saving external constants to {module_name}.ptd")
            executorch_program.write_tensor_data_to_file(args.outdir)


if __name__ == "__main__":
    main()
