# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Code generation for AXON delegate integration.

Handles subgraph naming, marker format, header symbol rewriting, and
generated table management. These are the generic parts needed by any
firmware project using the AXON ExecuTorch delegate.

Every call to ``AxonBackend.preprocess`` produces one AXON-compiled
subgraph. Each subgraph gets a **stable unique name** derived from a
hash of the intermediate binary, so the per-subgraph C symbols never
collide and rebuilding the same model gives the same names.

The generated directory layout::

    <axon_generated_dir>/
      axon_subgraph_<name>.h        -- one per delegated subgraph
      axon_subgraphs_table.h        -- regenerated on every preprocess() call
      .gitignore                    -- so this dir doesn't get committed
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Marker format ─────────────────────────────────────────────────
#
# The .pte's processed_bytes for an AXON delegate handle:
#
#   offset  size  field
#   ------  ----  -----
#   0       4     magic     "AXNG" (Axon NN, Generated)
#   4       4     version   little-endian uint32, currently 1
#   8       4     name_len  little-endian uint32, length of name in bytes
#   12      N     name      ASCII subgraph name, no NUL terminator
#   12+N    pad   pad to 4-byte alignment
#
# The total size is small (<256 bytes) and constant per subgraph.

_AXON_MARKER_MAGIC = b"AXNG"
_AXON_MARKER_VERSION = 1


def make_marker(subgraph_name: str) -> bytes:
    """Build the binary marker that goes into processed_bytes."""
    name_bytes = subgraph_name.encode("ascii")
    if len(name_bytes) > 255:
        raise ValueError(
            f"subgraph name too long ({len(name_bytes)} > 255 bytes): "
            f"{subgraph_name!r}"
        )
    payload = _AXON_MARKER_MAGIC
    payload += _AXON_MARKER_VERSION.to_bytes(4, "little")
    payload += len(name_bytes).to_bytes(4, "little")
    payload += name_bytes
    # Pad to 4-byte alignment so consumers can over-read safely.
    if len(payload) % 4:
        payload += b"\x00" * (4 - len(payload) % 4)
    return payload


# ── Subgraph naming ───────────────────────────────────────────────

# 12 hex chars = 48 bits of SHA-256 hash. At 48 bits, the probability
# of collision among N subgraphs is ~N^2 / 2^49 (birthday bound).
# For 1000 subgraphs: ~1.8e-9. Safe for any practical firmware build.
_NAME_HASH_LEN = 12


def derive_subgraph_name(model_name_prefix: str, intermediate_binary: bytes) -> str:
    """Stable unique subgraph name from the model name + binary content.

    The result is a valid C identifier and starts with the prefix so it's
    grep-friendly in the firmware build output.
    """
    digest = hashlib.sha256(intermediate_binary).hexdigest()[:_NAME_HASH_LEN]
    safe_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", model_name_prefix)
    return f"{safe_prefix}_{digest}"


# ── Header rewriting ──────────────────────────────────────────────

_RENAME_TOKEN_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def rewrite_header_symbols(header_text: str, old_name: str, new_name: str) -> str:
    """Rename all per-model C symbols in a Nordic-generated .h.

    The Nordic compiler embeds the model name into many C identifiers:

      - ``axon_model_const_<name>``
      - ``cmd_buffer_<name>``
      - ``model_<name>``
      - ``axon_model_<name>_packed_output_buf``
      - ``NRF_AXON_MODEL_<NAME>_PACKED_OUTPUT_SIZE`` (uppercase)
      - ``NRF_AXON_MODEL_<NAME>_MAX_IL_BUFFER_USED``
      - ``NRF_AXON_MODEL_<NAME>_MAX_PSUM_BUFFER_USED``
      - ``.model_name = "<name>"`` (a string literal)
    """
    if old_name == new_name:
        return header_text

    old_upper = old_name.upper()
    new_upper = new_name.upper()

    def replace_token(match: re.Match) -> str:
        tok = match.group(1)
        for old_marker, new_marker in (
            (f"axon_model_const_{old_name}", f"axon_model_const_{new_name}"),
            (f"axon_model_{old_name}_packed_output_buf",
             f"axon_model_{new_name}_packed_output_buf"),
            (f"NRF_AXON_MODEL_{old_upper}_PACKED_OUTPUT_SIZE",
             f"NRF_AXON_MODEL_{new_upper}_PACKED_OUTPUT_SIZE"),
            (f"NRF_AXON_MODEL_{old_upper}_MAX_IL_BUFFER_USED",
             f"NRF_AXON_MODEL_{new_upper}_MAX_IL_BUFFER_USED"),
            (f"NRF_AXON_MODEL_{old_upper}_MAX_PSUM_BUFFER_USED",
             f"NRF_AXON_MODEL_{new_upper}_MAX_PSUM_BUFFER_USED"),
            (f"cmd_buffer_{old_name}", f"cmd_buffer_{new_name}"),
            (f"model_{old_name}", f"model_{new_name}"),
        ):
            if tok == old_marker:
                return new_marker
        return tok

    rewritten = _RENAME_TOKEN_RE.sub(replace_token, header_text)

    # Also rewrite the .model_name string literal.
    rewritten = re.sub(
        r'\.model_name\s*=\s*"' + re.escape(old_name) + r'"',
        f'.model_name = "{new_name}"',
        rewritten,
    )
    return rewritten


def rewrite_op_extension_symbols(header_text: str) -> str:
    """Rewrite Nordic op extension symbols to generic names.

    Nordic's compiler generates ``nrf_axon_nn_op_extension_sigmoid`` etc.
    in the compiled headers. We rewrite these to ``axon_op_extension_*``
    so the firmware provides a consistent, non-vendor-prefixed interface.
    """
    replacements = {
        "nrf_axon_nn_op_extension_sigmoid": "axon_op_extension_sigmoid",
        "nrf_axon_nn_op_extension_tanh": "axon_op_extension_tanh",
        "nrf_axon_nn_op_extension_softmax": "axon_op_extension_softmax",
    }
    for old, new in replacements.items():
        header_text = header_text.replace(old, new)
    return header_text


# ── Generated directory layout ────────────────────────────────────

_TABLE_FILENAME = "axon_subgraphs_table.h"
_SUBGRAPH_PREFIX = "axon_subgraph_"


def write_subgraph_header(
    generated_dir: Path,
    subgraph_name: str,
    header_text: str,
) -> Path:
    """Write a single subgraph .h into the generated directory.

    Returns the path written. Creates ``generated_dir`` if missing.
    """
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_path = generated_dir / f"{_SUBGRAPH_PREFIX}{subgraph_name}.h"
    out_path.write_text(header_text)
    logger.info(f"AXON subgraph header -> {out_path}")
    return out_path


def regenerate_table(generated_dir: Path) -> Path:
    """Regenerate ``axon_subgraphs_table.h`` from the current contents.

    Scans ``generated_dir`` for ``axon_subgraph_*.h`` files and emits a
    deterministic master table. Idempotent: re-running with the same
    set of subgraph headers produces the same output bytes.
    """
    generated_dir.mkdir(parents=True, exist_ok=True)

    subgraph_paths = sorted(
        p for p in generated_dir.iterdir()
        if p.is_file() and p.name.startswith(_SUBGRAPH_PREFIX) and p.name.endswith(".h")
    )
    names = [p.name[len(_SUBGRAPH_PREFIX) : -2] for p in subgraph_paths]

    lines: list[str] = [
        "/* Auto-generated AXON subgraphs table — do NOT edit by hand.",
        " * Regenerated by the Nordic AXON backend on every AxonBackend.preprocess() call.",
        " * Owns the lookup from delegate marker name -> compiled model struct.",
        " */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        '#include "axon/nrf_axon_platform.h"',
        '#include "drivers/axon/nrf_axon_nn_infer.h"',
        "",
        "/* Each subgraph header allocates its own packed output buffer */",
        "#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1",
        "",
    ]
    for path in subgraph_paths:
        lines.append(f'#include "{path.name}"')
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char *name;")
    lines.append("    const nrf_axon_nn_compiled_model_s *model;")
    lines.append("} axon_subgraph_entry_t;")
    lines.append("")
    lines.append(f"#define AXON_SUBGRAPHS_COUNT {len(names)}")
    lines.append("")
    if names:
        lines.append("static const axon_subgraph_entry_t axon_subgraphs[] = {")
        for name in names:
            lines.append(f'    {{"{name}", &model_{name}}},')
        lines.append("};")
    else:
        lines.append("/* No subgraphs registered yet. */")
        lines.append("static const axon_subgraph_entry_t axon_subgraphs[1] = {{0}};")
    lines.append("")

    table_path = generated_dir / _TABLE_FILENAME
    table_path.write_text("\n".join(lines))
    logger.info(
        f"AXON subgraphs table -> {table_path} ({len(names)} subgraph(s))"
    )

    # Drop a tiny .gitignore so the directory is self-cleaning if it
    # ever gets committed by accident.
    gitignore = generated_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Auto-generated by AXON backend; do not commit.\n*\n!.gitignore\n")
    return table_path


def clean_generated_dir(generated_dir: Path) -> int:
    """Remove every ``axon_subgraph_*.h`` and the master table from a dir.

    Returns the number of files removed. Useful when re-exporting a
    different model and you want to drop the previous model's
    subgraphs from the firmware build.
    """
    if not generated_dir.exists():
        return 0
    removed = 0
    for path in generated_dir.iterdir():
        if path.is_file() and (
            path.name.startswith(_SUBGRAPH_PREFIX) or path.name == _TABLE_FILENAME
        ):
            path.unlink()
            removed += 1
    if removed:
        logger.info(f"Removed {removed} stale AXON file(s) from {generated_dir}")
    return removed
