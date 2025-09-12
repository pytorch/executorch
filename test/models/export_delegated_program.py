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
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
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
# Backends
#


@final
class StubBackend(BackendDetails):
    """No-op backend to test serialization/init."""

    @staticmethod
    def preprocess(*args, **kwargs) -> PreprocessResult:
        return PreprocessResult(processed_bytes=b"StubBackend:data")


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
        exported_program = export(WrapperModule(eager_module), args=inputs, strict=True)
    else:
        exported_program = export(eager_module, args=inputs, strict=True)

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
            exported_program = export(tagged_module, args=inputs, strict=True)
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            compile_config=edge_config,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch(config=et_config)
    else:
        edge: exir.EdgeProgramManager = to_edge(exported_program)
        lowered_module = to_backend(  # type: ignore[call-arg]
            backend_id,
            edge.exported_program(),
            # Just for the demo executor_backend.
            compile_specs=[CompileSpec(key="external_constants", value=b"")],
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
            export(composite_module, args=inputs, strict=True)
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
        outfile = os.path.join(args.outdir, f"{module_name}{suffix}.pte")
        executorch_program = export_module_to_program(
            module_class,
            backend_id=args.backend_id,
            extract_delegate_segments=not args.inline_delegate_segments,
            delegate_alignment=args.delegate_alignment,
            external_constants=args.external_constants,
        )
        with open(outfile, "wb") as fp:
            fp.write(executorch_program.buffer)
        print(f"Exported {module_name} and wrote program data to {outfile}")
        if args.external_constants:
            print(f"Saving external constants to {module_name}.ptd")
            executorch_program.write_tensor_data_to_file(args.outdir)


if __name__ == "__main__":
    main()
