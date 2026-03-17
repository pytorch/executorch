# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


class ETRecordAutoCollector:
    """Monkey-patch ETRecord APIs to auto-collect graph observations."""

    _installed: bool = False
    _originals: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def install(cls, collect_fn: Callable[[str, Any], None]) -> None:
        if cls._installed:
            return

        try:
            from executorch.devtools.etrecord._etrecord import ETRecord
        except Exception as exc:
            logging.warning("[Observatory] Failed to import ETRecord for auto-collect: %s", exc)
            return

        def _safe_collect(name: str, artifact: Any) -> None:
            try:
                collect_fn(name, artifact)
            except Exception as exc:
                logging.debug("[Observatory] Auto-collect skipped (%s): %s", name, exc)

        def _wrap_add_exported_program(original):
            def wrapped(self, exported_program):
                result = original(self, exported_program)
                if exported_program is None:
                    return result
                if isinstance(exported_program, dict):
                    for method_name, program in exported_program.items():
                        _safe_collect(f"ETRecord Exported/{method_name}", program)
                else:
                    _safe_collect("ETRecord Exported/forward", exported_program)
                return result

            return wrapped

        def _wrap_add_edge_dialect_program(original):
            def wrapped(self, edge_dialect_program):
                result = original(self, edge_dialect_program)
                processed = getattr(self, "edge_dialect_program", None)
                if isinstance(processed, dict):
                    for method_name, program in processed.items():
                        _safe_collect(f"ETRecord Edge/{method_name}", program)
                elif processed is not None:
                    _safe_collect("ETRecord Edge/forward", processed)
                return result

            return wrapped

        def _wrap_add_extra_export_modules(original):
            def wrapped(self, extra_recorded_export_modules):
                result = original(self, extra_recorded_export_modules)
                graph_map = getattr(self, "graph_map", {}) or {}
                for module_name, program in graph_map.items():
                    _safe_collect(f"ETRecord Extra/{module_name}", program)
                return result

            return wrapped

        patches = {
            "add_exported_program": _wrap_add_exported_program,
            "add_edge_dialect_program": _wrap_add_edge_dialect_program,
            "add_extra_export_modules": _wrap_add_extra_export_modules,
        }

        for method_name, wrap_builder in patches.items():
            original = getattr(ETRecord, method_name, None)
            if original is None:
                continue
            cls._originals[method_name] = original
            setattr(ETRecord, method_name, wrap_builder(original))

        cls._installed = True

    @classmethod
    def uninstall(cls) -> None:
        if not cls._installed:
            return

        try:
            from executorch.devtools.etrecord._etrecord import ETRecord
        except Exception:
            cls._originals.clear()
            cls._installed = False
            return

        for method_name, original in cls._originals.items():
            setattr(ETRecord, method_name, original)

        cls._originals.clear()
        cls._installed = False
