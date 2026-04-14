# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observatory CLI -- generic runner with shared helpers for backend CLIs.

Collection mode (default):
    python -m executorch.devtools.observatory \\
        [--output-html PATH] [--output-json PATH] SCRIPT [SCRIPT_ARGS...]

Visualize mode (JSON -> HTML, no script execution):
    python -m executorch.devtools.observatory visualize \\
        --input-json report.json --output-html report.html
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Shared helpers (importable by backend CLIs)
# ---------------------------------------------------------------------------


def make_collect_parser(prog=None):
    """Create the base argparse parser for collection mode.

    Backend CLIs can extend this with additional arguments before calling
    parse_args, e.g.::

        parser = make_collect_parser(prog="my-backend")
        parser.add_argument("--lense_recipe", choices=["accuracy"])
        args = parser.parse_args(sys.argv[1:])
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a script under Observatory and export reports.",
    )
    parser.add_argument("--output-html", default=None, help="HTML report output path")
    parser.add_argument("--output-json", default=None, help="JSON report output path")
    parser.add_argument("script", help="Script to run under Observatory")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the script",
    )
    return parser


def make_visualize_parser(prog=None):
    """Create the argparse parser for visualize mode."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate an HTML report from an existing JSON export.",
    )
    parser.add_argument("--input-json", required=True, help="Input JSON report path")
    parser.add_argument(
        "--output-html", required=True, help="Output HTML report path"
    )
    return parser


def run_visualize(input_json: str, output_html: str) -> None:
    if not os.path.isfile(input_json):
        logging.error("[Observatory CLI] Input JSON not found: %s", input_json)
        sys.exit(1)

    from .observatory import Observatory

    Observatory.clear()
    Observatory.generate_html_from_json(input_json, output_html)
    logging.info(
        "[Observatory CLI] visualize: html=%s from json=%s", output_html, input_json
    )


def run_observatory(
    script_path: str,
    script_argv: list[str],
    Observatory,
    output_html: str | None = None,
    output_json: str | None = None,
) -> None:
    """Shared run logic for all CLIs."""
    sys.argv = [script_path] + script_argv

    if output_html is None:
        output_html = "observatory_report.html"
    if output_json is None:
        if output_html.endswith(".html"):
            output_json = output_html[:-5] + ".json"
        else:
            output_json = "observatory_report.json"

    title = f"Observatory: {os.path.basename(script_path)}"

    try:
        with Observatory.enable_context(config={}):
            runpy.run_path(script_path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:
        logging.error("[Observatory CLI] Script raised: %s", exc)
    finally:
        os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        Observatory.export_json(output_json)
        Observatory.export_html_report(output_html, title=title, config={})
        collected = Observatory.list_collected()
        if collected:
            logging.info(
                "[Observatory CLI] Reports: html=%s json=%s (%d records: %s)",
                output_html,
                output_json,
                len(collected),
                ", ".join(collected),
            )
        else:
            logging.warning("[Observatory CLI] No records collected")


# ---------------------------------------------------------------------------
# Generic CLI entry point
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        parser = make_visualize_parser()
        args = parser.parse_args(sys.argv[2:])
        run_visualize(args.input_json, args.output_html)
        return

    parser = make_collect_parser()
    args = parser.parse_args(sys.argv[1:])

    from .observatory import Observatory
    from .lenses.pipeline_graph_collector import PipelineGraphCollectorLens

    Observatory.clear()
    Observatory.register_lens(PipelineGraphCollectorLens)

    run_observatory(
        args.script, args.script_args, Observatory, args.output_html, args.output_json
    )


if __name__ == "__main__":
    main()
