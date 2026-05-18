# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observatory CLI -- generic runner with shared helpers for backend CLIs.

Collection mode (default):
    python -m executorch.devtools.observatory \\
        [--output-html PATH] [--output-archive PATH] [--archive LABEL] \\
        SCRIPT [SCRIPT_ARGS...]

Visualize mode (Archive JSON -> HTML, no script execution):
    python -m executorch.devtools.observatory visualize \\
        --input-archive report.json --output-html report.html

Compare mode (overlay multiple Archives into one HTML report):
    python -m executorch.devtools.observatory compare \\
        --input-archive xnnpack.json --label XNNPACK/mv2 \\
        --input-archive qualcomm.json --label Qualcomm/mobilenet_v2 \\
        --output-html cross_backend.html [--title "..."]

``--archive`` sets ``Session.archive`` and -- when the user script does not
open its own outermost ``enter_context(region_name=...)`` -- also names the
session for the dashboard sidebar. Defaults to ``"default"``.
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
        parser.add_argument("--lens-recipe", action="append")
        args = parser.parse_args(sys.argv[1:])
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a script under Observatory and export reports.",
    )
    parser.add_argument("--output-html", default=None, help="HTML report output path")
    parser.add_argument(
        "--output-archive",
        default=None,
        dest="output_archive",
        help="Archive (JSON) output path.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        dest="archive",
        help=(
            "Archive label. Becomes Session.archive for every session this "
            "run produces and -- when the user script does not open a named "
            "outermost region -- the default session name as well. Drives "
            "compare-mode column grouping. Defaults to 'default'."
        ),
    )
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
        description="Generate an HTML report from an existing Archive (JSON).",
    )
    parser.add_argument(
        "--input-archive",
        required=True,
        dest="input_archive",
        help="Input Archive (JSON) path.",
    )
    parser.add_argument(
        "--output-html", required=True, help="Output HTML report path"
    )
    return parser


def make_compare_parser(prog=None):
    """Create the argparse parser for compare mode (multi-Archive overlay).

    Each ``--input-archive`` must be paired with a corresponding
    ``--label`` so the resulting Report (HTML) can group records under
    one top-level Region per archive (cf. RFC §4.5).
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Overlay multiple Archive (JSON) files into a single Report "
            "(HTML). Records, sessions, and region_stack values are "
            "prefixed with the corresponding --label so identically-named "
            "pipeline records (e.g. 'Annotated Model') stay distinct."
        ),
    )
    parser.add_argument(
        "--input-archive",
        action="append",
        required=True,
        dest="input_archives",
        help="Archive JSON to include (repeat for each archive).",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        dest="labels",
        help="Display label for the previous --input-archive (repeat per archive).",
    )
    parser.add_argument(
        "--output-html",
        required=True,
        help="Destination Report (HTML) path.",
    )
    parser.add_argument(
        "--title",
        default="Observatory Compare",
        help="Page title for the rendered HTML.",
    )
    return parser


def run_visualize(input_archive: str, output_html: str, *, setup_fn=None) -> None:
    if not os.path.isfile(input_archive):
        logging.error("[Observatory CLI] Input archive not found: %s", input_archive)
        sys.exit(1)

    from .observatory import Observatory

    Observatory.clear()
    if setup_fn is not None:
        setup_fn(Observatory)
    Observatory.generate_html_from_json(input_archive, output_html)
    logging.info(
        "[Observatory CLI] visualize: html=%s from archive=%s",
        output_html,
        input_archive,
    )


def _resolve_run_mode(script: str) -> tuple[str, str, str | None]:
    """Return (mode, target, pkg_root).

    mode='module': target is a dotted module name; pkg_root is prepended to sys.path.
    mode='script':  target is an absolute file path; pkg_root is None.

    Detection rules (in order):
    1. No '.py' suffix and no path separator → explicit dotted module name → 'module'
    2. File path whose directory has __init__.py → walk up to find package root → 'module'
    3. Everything else → 'script'
    """
    if not script.endswith(".py") and os.sep not in script and "/" not in script:
        return "module", script, None

    abs_path = os.path.abspath(script)
    script_dir = os.path.dirname(abs_path)

    if not os.path.isfile(os.path.join(script_dir, "__init__.py")):
        return "script", abs_path, None

    parts = [os.path.splitext(os.path.basename(abs_path))[0]]
    current = script_dir
    while True:
        parts.insert(0, os.path.basename(current))
        parent = os.path.dirname(current)
        if parent == current or not os.path.isfile(
            os.path.join(parent, "__init__.py")
        ):
            pkg_root = parent
            break
        current = parent

    return "module", ".".join(parts), pkg_root


def run_observatory(
    script_path: str,
    script_argv: list[str],
    Observatory,
    output_html: str | None = None,
    output_archive: str | None = None,
    archive: str | None = None,
) -> None:
    """Shared run logic for all CLIs."""
    sys.argv = [script_path] + script_argv

    if output_html is None:
        output_html = "observatory_report.html"
    if output_archive is None:
        if output_html.endswith(".html"):
            output_archive = output_html[:-5] + ".json"
        else:
            output_archive = "observatory_report.json"

    title = f"Observatory: {os.path.basename(script_path)}"
    mode, target, pkg_root = _resolve_run_mode(script_path)

    try:
        with Observatory.enable_context(config={}, archive=archive):
            if mode == "module":
                if pkg_root is not None and pkg_root not in sys.path:
                    sys.path.insert(0, pkg_root)
                logging.info("[Observatory CLI] Running as module: %s", target)
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            else:
                logging.info("[Observatory CLI] Running as script: %s", target)
                runpy.run_path(target, run_name="__main__")
    except SystemExit:
        pass
    except ImportError as exc:
        logging.error("[Observatory CLI] Import error in '%s': %s", script_path, exc)
        if "relative import" in str(exc) or "attempted relative import" in str(exc):
            logging.error(
                "  Hint: this script uses relative imports and must run as a Python module.\n"
                "  Option A — ensure the script's directory contains __init__.py so it is\n"
                "             auto-detected as a package member.\n"
                "  Option B — pass a dotted module name instead of a file path:\n"
                "             python -m executorch.devtools.observatory"
                " <package>.<subpackage>.<module> [args...]"
            )
        elif mode == "module":
            logging.error(
                "  Hint: module '%s' could not be imported.\n"
                "  Ensure the package root '%s' is on PYTHONPATH or the package is installed.",
                target,
                pkg_root or "unknown",
            )
    except Exception as exc:
        logging.error(
            "[Observatory CLI] '%s' raised: %s  (run mode: %s, target: %s)",
            os.path.basename(script_path),
            exc,
            mode,
            target,
        )
    finally:
        os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_archive) or ".", exist_ok=True)
        Observatory.export_json(output_archive)
        Observatory.export_html_report(output_html, title=title, config={})
        collected = Observatory.list_collected()
        if collected:
            logging.info(
                "[Observatory CLI] Reports: html=%s archive=%s (%d records: %s)",
                output_html,
                output_archive,
                len(collected),
                ", ".join(collected),
            )
        else:
            logging.warning("[Observatory CLI] No records collected")


# ---------------------------------------------------------------------------
# Generic CLI entry point
# ---------------------------------------------------------------------------


def run_compare(
    input_archives: list[str],
    labels: list[str],
    output_html: str,
    title: str = "Observatory Compare",
    *,
    setup_fn=None,
) -> None:
    """Load multiple Archives and render a side-by-side Report (HTML)."""
    if len(input_archives) != len(labels):
        logging.error(
            "[Observatory CLI] compare: %d --input-archive values but %d --label values"
            " (must match 1:1).",
            len(input_archives),
            len(labels),
        )
        sys.exit(1)
    for path in input_archives:
        if not os.path.isfile(path):
            logging.error("[Observatory CLI] compare: archive not found: %s", path)
            sys.exit(1)

    from .observatory import Observatory

    Observatory.clear()
    if setup_fn is not None:
        setup_fn(Observatory)
    Observatory.compare_archives(
        archive_paths=input_archives,
        labels=labels,
        html_path=output_html,
        title=title,
    )
    logging.info(
        "[Observatory CLI] compare: html=%s from %d archives",
        output_html,
        len(input_archives),
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        parser = make_visualize_parser()
        args = parser.parse_args(sys.argv[2:])
        run_visualize(args.input_archive, args.output_html)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = make_compare_parser()
        args = parser.parse_args(sys.argv[2:])
        run_compare(
            input_archives=args.input_archives,
            labels=args.labels,
            output_html=args.output_html,
            title=args.title,
        )
        return

    parser = make_collect_parser()
    args = parser.parse_args(sys.argv[1:])

    from .observatory import Observatory
    from .lenses.pipeline_graph_collector import PipelineGraphCollectorLens

    Observatory.clear()
    Observatory.register_lens(PipelineGraphCollectorLens)

    run_observatory(
        args.script,
        args.script_args,
        Observatory,
        args.output_html,
        args.output_archive,
        archive=args.archive,
    )


if __name__ == "__main__":
    main()
