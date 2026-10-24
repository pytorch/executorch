# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Add or verify the BSD-3-Clause license header on every source and build file
of the ESP-NN backend and its examples.

    python examples/espressif/scripts/license_headers.py --holder "Your Name or Company"
    python examples/espressif/scripts/license_headers.py --check

The header matches the ExecuTorch convention (contributor copyright line plus
the reference to the BSD-style LICENSE at the repository root) and adds an
SPDX identifier. Re-running with a different --holder replaces the line.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPES = [ROOT / "backends" / "espressif", ROOT / "examples" / "espressif"]
DEFAULT_HOLDER = "Mwaniki Njagi"
YEAR = "2026"
HASH_STYLE = {".py", ".sh", ".cmake", ".yml", ".yaml"}
HASH_NAMES = {"CMakeLists.txt", "Kconfig", "Kconfig.projbuild"}
BLOCK_STYLE = {".c", ".cc", ".cpp", ".h", ".hpp", ".S"}
SKIP_PARTS = {
    "build",
    "managed_components",
    "__pycache__",
    "benchmarks",
    "bundles",
    "models",
}
MARKER = "SPDX-License-Identifier: BSD-3-Clause"
BODY = [
    "This source code is licensed under the BSD-style license found in the",
    "LICENSE file in the root directory of this source tree.",
]


def header_lines(holder: str) -> list[str]:
    return [
        f"Copyright {YEAR} {holder}.",
        "All rights reserved.",
        "",
        *BODY,
        "",
        MARKER,
    ]


def render(style: str, holder: str) -> str:
    lines = header_lines(holder)
    if style == "hash":
        return "\n".join(f"# {line}".rstrip() for line in lines) + "\n"
    return "/*\n" + "\n".join(f" * {line}".rstrip() for line in lines) + "\n */\n"


def style_for(path: Path) -> str | None:
    if path.name in HASH_NAMES or path.suffix in HASH_STYLE:
        return "hash"
    if path.suffix in BLOCK_STYLE:
        return "block"
    return None


def strip_existing(text: str, style: str) -> str:
    if MARKER not in text.split("\n\n", 2)[0] + text[:600]:
        return text
    if style == "hash":
        pattern = re.compile(r"\A(#[^\n]*\n)+?# " + re.escape(MARKER) + r"\n\n?")
    else:
        pattern = re.compile(r"\A/\*.*?" + re.escape(MARKER) + r"\n \*/\n\n?", re.S)
    return pattern.sub("", text, count=1)


def apply(path: Path, holder: str) -> bool:
    style = style_for(path)
    text = path.read_text()
    shebang = ""
    if text.startswith("#!"):
        shebang, _, text = text.partition("\n")
        shebang += "\n"
    body = strip_existing(text, style)
    body = body.lstrip("\n")
    updated = shebang + render(style, holder) + ("\n" + body if body else "")
    if updated != shebang + text:
        path.write_text(updated)
        return True
    return False


def files():
    for scope in SCOPES:
        for path in sorted(scope.rglob("*")):
            parts = path.relative_to(scope).parts[:-1]
            if path.is_file() and not any(
                p in SKIP_PARTS or p.startswith("build") for p in parts
            ):
                if style_for(path):
                    yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--holder", default=DEFAULT_HOLDER)
    parser.add_argument(
        "--check", action="store_true", help="fail if any file lacks the header"
    )
    args = parser.parse_args()
    missing = []
    changed = 0
    for path in files():
        if args.check:
            if MARKER not in path.read_text()[:800]:
                missing.append(path)
        elif apply(path, args.holder):
            changed += 1
    if args.check:
        for path in missing:
            print(f"missing license header: {path.relative_to(ROOT)}")
        print(f"{len(missing)} file(s) without a header")
        return 1 if missing else 0
    print(f"updated {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
