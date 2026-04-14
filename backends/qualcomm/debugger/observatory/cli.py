# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qualcomm Observatory CLI — QNN-specific lens configuration.

Graph collection only (default):
    python -m executorch.backends.qualcomm.debugger.observatory SCRIPT [ARGS...]

With accuracy debugging:
    python -m executorch.backends.qualcomm.debugger.observatory --accuracy SCRIPT [ARGS...]

Visualize mode (JSON -> HTML):
    python -m executorch.backends.qualcomm.debugger.observatory visualize \\
        --input report.json --output report.html [--title "My Report"]
"""

from __future__ import annotations

import sys


def main():
    from executorch.devtools.observatory.cli import (
        parse_observatory_args,
        run_observatory,
        run_visualize,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        run_visualize()
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: python -m executorch.backends.qualcomm.debugger.observatory "
            "[--accuracy] [--json-only] [--no-report] "
            "[--report-title TITLE] [--report-dir DIR] "
            "[--report-html PATH] [--report-json PATH] "
            "SCRIPT [SCRIPT_ARGS...]"
        )
        sys.exit(0)

    usage = (
        "Usage: python -m executorch.backends.qualcomm.debugger.observatory "
        "[--accuracy] [--json-only] [--no-report] "
        "[--report-title TITLE] [--report-dir DIR] "
        "[--report-html PATH] [--report-json PATH] "
        "SCRIPT [SCRIPT_ARGS...]"
    )
    obs_flags, script_path, script_argv = parse_observatory_args(usage_str=usage)

    from executorch.devtools.observatory.observatory import Observatory
    from executorch.devtools.observatory.lenses.pipeline_graph_collector import (
        PipelineGraphCollectorLens,
    )

    from .lenses.qnn_patches import install_qnn_patches

    Observatory.clear()

    PipelineGraphCollectorLens.register_backend_patches(install_qnn_patches)
    Observatory.register_lens(PipelineGraphCollectorLens)

    if obs_flags["--accuracy"]:
        from executorch.devtools.observatory.lenses.accuracy import AccuracyLens

        from .lenses.qnn_dataset_patches import install_qnn_dataset_patches

        AccuracyLens.register_dataset_patches(install_qnn_dataset_patches)
        Observatory.register_lens(AccuracyLens)

        from .lenses.per_layer_accuracy import PerLayerAccuracyLens
        cls.register_lens(PerLayerAccuracyLens)



    run_observatory(obs_flags, script_path, script_argv, Observatory)


if __name__ == "__main__":
    main()
