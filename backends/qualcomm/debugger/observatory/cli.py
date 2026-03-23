# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observatory CLI runner — zero-config debugging for ExecuTorch scripts.

Usage:
    python -m executorch.backends.qualcomm.debugger.observatory.cli \\
        examples/qualcomm/oss_scripts/swin_v2_t.py \\
        --model SM8650 -b ./build-android -c -d imagenet-mini/val -a ./swin_v2_t

The runner wraps the target script in an Observatory context, automatically
collecting graph snapshots and accuracy metrics at each compilation stage.
"""

from __future__ import annotations

import logging
import os
import runpy
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_observatory_args():
    """Extract observatory-specific flags from sys.argv, return (obs_args, script_argv)."""

    obs_flags = {
        "--no-accuracy": False,
        "--no-report": False,
        "--report-title": None,
    }
    script_path = None
    script_argv = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--no-accuracy":
            obs_flags["--no-accuracy"] = True
        elif arg == "--no-report":
            obs_flags["--no-report"] = True
        elif arg == "--report-title" and i + 1 < len(sys.argv):
            i += 1
            obs_flags["--report-title"] = sys.argv[i]
        elif script_path is None and not arg.startswith("--"):
            script_path = arg
        else:
            script_argv.append(arg)
        i += 1

    if script_path is None:
        print("Usage: python -m executorch.backends.qualcomm.debugger.observatory.cli "
              "[--no-accuracy] [--no-report] [--report-title TITLE] SCRIPT [SCRIPT_ARGS...]")
        sys.exit(1)

    return obs_flags, script_path, script_argv


def main():
    obs_flags, script_path, script_argv = _parse_observatory_args()

    from .observatory import Observatory
    from .lenses.pipeline_graph_collector import PipelineGraphCollectorLens

    Observatory.clear()
    Observatory.register_lens(PipelineGraphCollectorLens)

    if not obs_flags["--no-accuracy"]:
        from .lenses.accuracy import AccuracyLens

        Observatory.register_lens(AccuracyLens)

    sys.argv = [script_path] + script_argv

    config: dict = {}
    artifact_path = "."

    # Try to extract artifact path from script args for report output
    for j, arg in enumerate(script_argv):
        if arg in ("-a", "--artifact") and j + 1 < len(script_argv):
            artifact_path = script_argv[j + 1]
            break

    title = obs_flags["--report-title"] or f"Observatory: {os.path.basename(script_path)}"

    try:
        with Observatory.enable_context(config=config):
            runpy.run_path(script_path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:
        logging.error("[Observatory CLI] Script raised: %s", exc)
    finally:
        if not obs_flags["--no-report"]:
            os.makedirs(artifact_path, exist_ok=True)
            report_html = os.path.join(artifact_path, "observatory_report.html")
            report_json = os.path.join(artifact_path, "observatory_report.json")
            Observatory.export_html_report(report_html, title=title, config=config)
            Observatory.export_json(report_json)
            collected = Observatory.list_collected()
            if collected:
                logging.info(
                    "[Observatory CLI] Report: %s (%d records: %s)",
                    report_html,
                    len(collected),
                    ", ".join(collected),
                )
            else:
                logging.warning("[Observatory CLI] No records collected")


if __name__ == "__main__":
    main()
