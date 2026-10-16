# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchvision
from executorch.backends.cpu.partitioner import CPU_DELEGATE_VERSION
from executorch.backends.cpu.recipes import CPURecipeType
from executorch.backends.native.serialization import deserialize_graph
from executorch.backends.native.serialization.schema import OpKind
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack.recipes import XNNPackRecipeType
from executorch.export import export, ExportRecipe, ExportSession
from torchvision.models import get_model, get_model_weights, list_models


logger: logging.Logger = logging.getLogger(__name__)
RECIPE_TYPES = {
    "cpu": CPURecipeType.FP32,
    "xnnpack": XNNPackRecipeType.FP32,
}
DEFAULT_MODEL = "convnext_tiny"
DEFAULT_IMAGE_SIZE = 224
IMAGE_SIZE_OVERRIDES = {"inception_v3": 299}


def classification_models() -> list[str]:
    return sorted(list_models(torchvision.models))


def image_size_for(
    model: str, override: int | None, *, random_weights: bool = False
) -> int:
    if override is not None:
        return override
    if not random_weights:
        return get_model_weights(model).DEFAULT.transforms().crop_size[0]
    return IMAGE_SIZE_OVERRIDES.get(model, DEFAULT_IMAGE_SIZE)


def synthetic_image(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.arange(math.prod(shape)) % 251).float().div(250).reshape(shape)


def capture(
    args: argparse.Namespace,
) -> tuple[torch.export.ExportedProgram, torch.Tensor]:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    destination = args.output / f"{args.model}.pt2"
    if args.pt2:
        exported = torch.export.load(args.pt2)
        input_name = exported.graph_signature.user_inputs[0]
        input_node = next(
            node for node in exported.graph.nodes if node.name == input_name
        )
        image = synthetic_image(tuple(input_node.meta["val"].shape))
        if args.pt2.resolve() != destination.resolve():
            shutil.copyfile(args.pt2, destination)
    else:
        size = image_size_for(
            args.model, args.image_size, random_weights=args.random_weights
        )
        weights = None if args.random_weights else get_model_weights(args.model).DEFAULT
        model_kwargs = {"image_size": size} if args.model.startswith("vit_") else {}
        model = get_model(args.model, weights=weights, **model_kwargs).eval()
        image = synthetic_image((1, 3, size, size))
        exported = torch.export.export(model, (image,))
        torch.export.save(exported, destination)
    with torch.no_grad():
        exported.module()(image).contiguous().numpy().tofile(args.output / "eager.bin")
    image.numpy().tofile(args.output / "input.bin")
    return exported, image


def write_manifest(
    args: argparse.Namespace,
    spec: dict[str, Any],
    exported: torch.export.ExportedProgram,
    recipe: ExportRecipe,
    session: ExportSession,
    image: torch.Tensor,
) -> None:
    is_cpu = spec["backend"] == "cpu"
    passes = [] if is_cpu else XNNPACKPassManager(exported).passes
    files = {}
    artifact = "model.pte"
    for name in (f"{args.model}.pt2", "input.bin", "eager.bin", "golden.bin", artifact):
        path = args.output / name
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        files[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    program = session.get_executorch_program()
    semantic_nodes = None
    if is_cpu:
        semantic_nodes = [
            node.name
            for delegate in program.execution_plan[0].delegates
            if delegate.id == "CpuBackend"
            for node in deserialize_graph(
                bytes(program.backend_delegate_data[delegate.processed.index].data)
            ).nodes
            if node.op_kind == OpKind.CALL_FUNCTION
        ]
    manifest = {
        "version": 1,
        "model": args.model,
        "runtime": "et",
        "artifact": artifact,
        "preprocess": "CpuBackend" if is_cpu else "XnnpackBackend",
        "source_revision": args.source_revision,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "recipe": spec,
        "quantization": None,
        "cpu_delegate_version": CPU_DELEGATE_VERSION if is_cpu else None,
        "recipe_passes": {"aten": [], "edge": [], "edge_manager": []},
        "backend_default_passes": [f"{p.__module__}.{p.__qualname__}" for p in passes],
        "preserved_ops": sorted(
            {
                str(op)
                for partitioner in recipe.lowering_recipe.partitioners
                for op in partitioner.ops_to_not_decompose(exported)[0]
            }
        ),
        "stages": [stage.name for stage in session.get_stage_artifacts()],
        "delegates": dict(Counter(d.id for d in program.execution_plan[0].delegates)),
        "semantic_nodes": semantic_nodes,
        "input": {
            "shape": list(image.shape),
            "dtype": "float32",
            "pattern": "(index % 251) / 250",
        },
        "files": files,
    }
    for name, value in (("recipe.json", spec), ("export.json", manifest)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=tuple(RECIPE_TYPES), default="cpu")
    parser.add_argument("--pt2", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--source-revision")
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported torchvision classification models and exit",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.list_models:
        for name in classification_models():
            print(name)
        return
    if args.output is None or args.source_revision is None:
        parser.error("--output and --source-revision are required")
    if args.pt2:
        if args.random_weights or args.image_size is not None:
            parser.error("--random-weights and --image-size cannot be used with --pt2")
        args.model = args.model or args.pt2.stem
    else:
        args.model = args.model or DEFAULT_MODEL
        if args.model not in classification_models():
            parser.error(f"Unknown torchvision model {args.model!r}; see --list-models")
    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    recipe_type = RECIPE_TYPES[args.backend]
    spec = {
        "version": 1,
        "backend": args.backend,
        "recipe": recipe_type.value,
        "kwargs": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    exported, image = capture(args)
    recipe = ExportRecipe.get_recipe(recipe_type)
    session = export(exported, export_recipe=recipe)
    edge_module = session.get_edge_program_manager().exported_program().module()
    with torch.no_grad():
        edge_module(image).numpy().tofile(args.output / "golden.bin")
    artifact = args.output / "model.pte"
    artifact.write_bytes(session.get_pte_buffer())
    write_manifest(args, spec, exported, recipe, session, image)
    logger.info("Exported %s to %s", spec["recipe"], artifact)


if __name__ == "__main__":
    main()
