# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observatory CLI — run mode and visualize mode.

Run mode (default):
    python -m backends.qualcomm.debugger.observatory.cli \\
        [--no-accuracy] [--json-only] [--report-title TITLE] \\
        [--report-dir DIR] [--report-html PATH] [--report-json PATH] \\
        SCRIPT [SCRIPT_ARGS...]

Visualize mode (JSON -> HTML, no script execution):
    python -m backends.qualcomm.debugger.observatory.cli visualize \\
        --input report.json --output report.html [--title "My Report"]

See backends/qualcomm/debugger/observatory/USAGE.md for full workflow docs.
"""

from __future__ import annotations

import logging
import os
import runpy
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _usage_run() -> str:
    return (
        "Usage: python -m backends.qualcomm.debugger.observatory.cli "
        "[--no-accuracy] [--json-only] [--no-report] "
        "[--report-title TITLE] [--report-dir DIR] "
        "[--report-html PATH] [--report-json PATH] "
        "SCRIPT [SCRIPT_ARGS...]"
    )


def _usage_visualize() -> str:
    return (
        "Usage: python -m backends.qualcomm.debugger.observatory.cli visualize "
        "--input report.json --output report.html [--title TITLE]"
    )


def _read_value(argv: list[str], index: int, flag_name: str) -> tuple[str, int]:
    """Read next token as value for a flag and return (value, next_index)."""
    if index + 1 >= len(argv):
        print(f"Missing value for {flag_name}")
        print(_usage_run())
        sys.exit(1)
    return argv[index + 1], index + 1


def _parse_visualize_args() -> tuple[str, str, str]:
    """Parse argv for: cli visualize --input X --output Y [--title T]"""
    argv = sys.argv[2:]  # skip 'visualize' token
    input_path = None
    output_path = None
    title = "Observatory Report"

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--input", "-i"):
            input_path, i = _read_value(argv, i, arg)
        elif arg.startswith("--input="):
            input_path = arg.split("=", 1)[1]
        elif arg in ("--output", "-o"):
            output_path, i = _read_value(argv, i, arg)
        elif arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif arg == "--title":
            title, i = _read_value(argv, i, arg)
        elif arg.startswith("--title="):
            title = arg.split("=", 1)[1]
        elif arg in ("-h", "--help"):
            print(_usage_visualize())
            sys.exit(0)
        else:
            print(f"Unknown flag for visualize: {arg}")
            print(_usage_visualize())
            sys.exit(1)
        i += 1

    if not input_path or not output_path:
        print("visualize requires --input and --output")
        print(_usage_visualize())
        sys.exit(1)

    return input_path, output_path, title


def _run_visualize() -> None:
    input_path, output_path, title = _parse_visualize_args()

    if not os.path.isfile(input_path):
        logging.error("[Observatory CLI] Input JSON not found: %s", input_path)
        sys.exit(1)

    from .observatory import Observatory

    Observatory.clear()
    Observatory.generate_html_from_json(input_path, output_path, title=title)
    logging.info(
        "[Observatory CLI] visualize: html=%s from json=%s", output_path, input_path
    )


def _parse_observatory_args():
    """Extract observatory flags from argv before SCRIPT, then pass all remaining args through."""

    obs_flags = {
        "--no-accuracy": False,
        "--no-report": False,
        "--json-only": False,
        "--report-title": None,
        "--report-dir": None,
        "--report-html": None,
        "--report-json": None,
    }
    script_path = None
    script_argv = []
    argv = list(sys.argv[1:])

    i = 0
    while i < len(argv):
        arg = argv[i]
        if script_path is not None:
            script_argv.append(arg)
            i += 1
            continue

        if arg == "--":
            if i + 1 >= len(argv):
                print(_usage_run())
                sys.exit(1)
            script_path = argv[i + 1]
            script_argv.extend(argv[i + 2 :])
            break
        if arg == "--no-accuracy":
            obs_flags["--no-accuracy"] = True
        elif arg == "--no-report":
            obs_flags["--no-report"] = True
        elif arg == "--json-only":
            obs_flags["--json-only"] = True
        elif arg in ("--report-title", "--obs-report-title"):
            value, i = _read_value(argv, i, arg)
            obs_flags["--report-title"] = value
        elif arg.startswith("--report-title=") or arg.startswith("--obs-report-title="):
            obs_flags["--report-title"] = arg.split("=", 1)[1]
        elif arg in ("--report-dir", "--obs-report-dir"):
            value, i = _read_value(argv, i, arg)
            obs_flags["--report-dir"] = value
        elif arg.startswith("--report-dir=") or arg.startswith("--obs-report-dir="):
            obs_flags["--report-dir"] = arg.split("=", 1)[1]
        elif arg in ("--report-html", "--obs-report-html"):
            value, i = _read_value(argv, i, arg)
            obs_flags["--report-html"] = value
        elif arg.startswith("--report-html=") or arg.startswith("--obs-report-html="):
            obs_flags["--report-html"] = arg.split("=", 1)[1]
        elif arg in ("--report-json", "--obs-report-json"):
            value, i = _read_value(argv, i, arg)
            obs_flags["--report-json"] = value
        elif arg.startswith("--report-json=") or arg.startswith("--obs-report-json="):
            obs_flags["--report-json"] = arg.split("=", 1)[1]
        elif arg.startswith("-"):
            print(f"Unknown observatory flag before SCRIPT: {arg}")
            print(_usage_run())
            sys.exit(1)
        else:
            script_path = arg
        i += 1

    if script_path is None:
        print(_usage_run())
        sys.exit(1)

    return obs_flags, script_path, script_argv


def _infer_report_dir(script_argv: list[str]) -> str:
    """Infer output directory from target script args (artifact/output_dir), else cwd."""
    keys = ("-a", "--artifact", "-o", "--output_dir")
    for idx, arg in enumerate(script_argv):
        if arg in keys and idx + 1 < len(script_argv):
            return script_argv[idx + 1]
        if arg.startswith("--artifact="):
            return arg.split("=", 1)[1]
        if arg.startswith("--output_dir="):
            return arg.split("=", 1)[1]
    return "."


def _resolve_report_paths(obs_flags: dict[str, object], script_argv: list[str]) -> tuple[str, str]:
    report_dir = obs_flags["--report-dir"] or _infer_report_dir(script_argv)
    report_html = obs_flags["--report-html"]
    report_json = obs_flags["--report-json"]

    if report_html is None:
        report_html = os.path.join(str(report_dir), "observatory_report.html")
    if report_json is None:
        if report_html.endswith(".html"):
            report_json = report_html[:-5] + ".json"
        else:
            report_json = os.path.join(str(report_dir), "observatory_report.json")

    return str(report_html), str(report_json)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        _run_visualize()
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(_usage_run())
        print()
        print(_usage_visualize())
        sys.exit(0)

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
    report_html, report_json = _resolve_report_paths(obs_flags, script_argv)

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
            os.makedirs(os.path.dirname(report_html) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(report_json) or ".", exist_ok=True)
            Observatory.export_json(report_json)
            collected = Observatory.list_collected()
            if not obs_flags["--json-only"]:
                Observatory.export_html_report(report_html, title=title, config=config)
            if collected:
                if obs_flags["--json-only"]:
                    logging.info(
                        "[Observatory CLI] Reports: json=%s (%d records: %s)",
                        report_json,
                        len(collected),
                        ", ".join(collected),
                    )
                else:
                    logging.info(
                        "[Observatory CLI] Reports: html=%s json=%s (%d records: %s)",
                        report_html,
                        report_json,
                        len(collected),
                        ", ".join(collected),
                    )
            else:
                logging.warning("[Observatory CLI] No records collected")


if __name__ == "__main__":
    main()
