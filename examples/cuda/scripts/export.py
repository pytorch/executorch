# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer with CUDA delegate.

import argparse
import pathlib

import torch

from executorch.backends.cuda.cuda_backend import CudaBackend

from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.extension.export_util.utils import save_pte_program

# Script to export a model with CUDA delegation.

_EDGE_COMPILE_CONFIG = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): enable dim_order in backend
)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("./"),
        help="Output directory for the exported model",
    )
    parser.add_argument("--generate_etrecord", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_processed_bytes", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--cuda_include_ptx",
        choices=("ON", "OFF"),
        help="Explicitly enable or disable PTX in the exported CUDA AOTI library",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed model initialization and example input generation",
    )
    parser.add_argument(
        "--save_example_io",
        action="store_true",
        help="Also write the example inputs and the eager outputs as raw files "
        "(<model>_input_<i>.bin, <model>_expected_<i>.bin) for executor_runner "
        "--inputs and --output_file",
    )

    args = parser.parse_args()
    return args


def save_processed_bytes(processed_bytes, base_name: str):
    filename = f"{base_name}.bin"
    print(f"Saving processed bytes to {filename}")
    with open(filename, "wb") as file:
        file.write(processed_bytes)
    return


def save_example_io(model, example_args, base_name: str, output_dir: pathlib.Path):
    with torch.no_grad():
        outputs = model(*example_args)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, tensors in (("input", example_args), ("expected", outputs)):
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError(
                f"--save_example_io writes tensors only; {base_name} has {kind}s of "
                f"types {[type(t).__name__ for t in tensors]}"
            )
        for i, tensor in enumerate(tensors):
            path = output_dir / f"{base_name}_{kind}_{i}.bin"
            path.write_bytes(
                tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            )


def main():
    args = parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    (
        model,
        example_args,
        example_kwargs,
        dynamic_shapes,
    ) = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[args.model_name])
    model = model.eval()
    if args.save_example_io:
        save_example_io(model, example_args, args.model_name, args.output_dir)
    exported_programs = torch.export.export(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )

    compile_specs = [CudaBackend.generate_method_name_compile_spec(args.model_name)]
    if args.cuda_include_ptx is not None:
        compile_specs.append(
            CompileSpec("cuda_include_ptx", args.cuda_include_ptx.encode())
        )
    partitioner = CudaPartitioner(compile_specs)

    et_prog = to_edge_transform_and_lower(
        exported_programs,
        partitioner=[partitioner],
        compile_config=_EDGE_COMPILE_CONFIG,
        generate_etrecord=args.generate_etrecord,
    )
    exec_program = et_prog.to_executorch()
    save_pte_program(exec_program, args.model_name, args.output_dir)
    if args.generate_etrecord:
        exec_program.get_etrecord().save(f"{args.model_name}_etrecord.bin")


if __name__ == "__main__":
    main()
