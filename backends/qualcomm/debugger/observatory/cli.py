# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qualcomm Observatory CLI -- QNN-specific lens configuration.

Collection mode (default):
    python -m executorch.backends.qualcomm.debugger.observatory \\
        [--output-html PATH] [--output-json PATH] SCRIPT [SCRIPT_ARGS...]

With one or more lens recipes (repeat the flag or comma-separate):
    python -m executorch.backends.qualcomm.debugger.observatory \\
        --lens-recipe adb --lens-recipe accuracy SCRIPT [SCRIPT_ARGS...]
    python -m executorch.backends.qualcomm.debugger.observatory \\
        --lens-recipe adb,accuracy SCRIPT [SCRIPT_ARGS...]

Visualize mode (JSON -> HTML):
    python -m executorch.backends.qualcomm.debugger.observatory visualize \\
        --input-json report.json --output-html report.html
"""

from __future__ import annotations

import sys


_RECIPE_CHOICES = ["accuracy", "adb"]


def _parse_recipe_values(values):
    """Split repeated and comma-separated --lens-recipe values."""
    recipes: set = set()
    for v in values or []:
        for token in str(v).split(","):
            token = token.strip()
            if not token:
                continue
            if token not in _RECIPE_CHOICES:
                raise SystemExit(
                    "argument --lens-recipe: invalid choice: "
                    f"{token!r} (choose from {', '.join(repr(c) for c in _RECIPE_CHOICES)})"
                )
            recipes.add(token)
    return recipes


def main():
    from executorch.devtools.observatory.cli import (
        make_collect_parser,
        make_visualize_parser,
        run_observatory,
        run_visualize,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        parser = make_visualize_parser()
        args = parser.parse_args(sys.argv[2:])
        run_visualize(args.input_json, args.output_html)
        return

    parser = make_collect_parser(
        prog="python -m executorch.backends.qualcomm.debugger.observatory"
    )
    parser.add_argument(
        "--lens-recipe",
        action="append",
        default=[],
        metavar="RECIPE",
        help=(
            "Lens recipe to enable. Repeat the flag or comma-separate to "
            "enable multiple (e.g. '--lens-recipe adb --lens-recipe accuracy' "
            "or '--lens-recipe adb,accuracy'). "
            f"Choices: {', '.join(_RECIPE_CHOICES)}"
        ),
    )
    args = parser.parse_args(sys.argv[1:])
    recipes = _parse_recipe_values(args.lens_recipe)

    from executorch.devtools.observatory.observatory import Observatory
    from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
        PipelineGraphCollectorLens,
    )
    from .lenses.qnn_patches import install_qnn_patches

    Observatory.clear()
    PipelineGraphCollectorLens.register_backend_patches(install_qnn_patches)
    Observatory.register_lens(PipelineGraphCollectorLens)

    if "accuracy" in recipes:
        from executorch.devtools.observatory.lenses.accuracy import AccuracyLens
        from .lenses.qnn_dataset_patches import install_qnn_dataset_patches

        AccuracyLens.register_dataset_patches(install_qnn_dataset_patches)
        Observatory.register_lens(AccuracyLens)

        from executorch.devtools.observatory.lenses.per_layer_accuracy import (
            PerLayerAccuracyLens,
        )

        Observatory.register_lens(PerLayerAccuracyLens)

    if "adb" in recipes:
        from .lenses.adb import AdbLens
        from .lenses.adb_patches import install_adb_patches

        AdbLens.register_adb_patches(install_adb_patches)
        Observatory.register_lens(AdbLens)

    run_observatory(
        args.script, args.script_args, Observatory, args.output_html, args.output_json
    )


if __name__ == "__main__":
    main()
