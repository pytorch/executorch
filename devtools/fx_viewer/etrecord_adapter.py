# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter: ETRecord bundle -> FXGraphCompareExporter.

Produces a ready-to-export compare view from an ``.etrecord`` file. Handles:

  * Missing ``from_node`` / ``debug_handle`` on the raw aten stored program
    by re-tracing via ``run_decompositions({})`` and running
    ``DebugHandleGeneratorPass``. The two enrichments are checked
    independently and applied only when absent on every call/op node.
  * Optional backend overlay on the edge graph, built by cross-joining
    ``_debug_handle_map`` (instruction -> handles) with ``_delegate_map``
    (instruction -> backend name).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Union

import torch.fx

from executorch.devtools.etrecord import ETRecord, parse_etrecord
from executorch.exir.passes.debug_handle_generator_pass import DebugHandleGeneratorPass

from .color_rules import CategoricalColorRule
from .compare_exporter import FXGraphCompareExporter
from .exporter import FXGraphExporter
from .extension import GraphExtension


def _has_meta_key(gm: torch.fx.GraphModule, key: str) -> bool:
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        v = n.meta.get(key)
        if v is not None and v != 0:
            return True
    return False


def _enrich_missing_meta(ep):
    """Populate ``from_node`` / ``debug_handle`` if absent on every call/op node.

    Independent checks per key. ``run_decompositions({})`` re-traces the
    program through the ``PropagateUnbackedSymInts`` interpreter, which sets
    ``from_node`` on every emitted node. ``DebugHandleGeneratorPass`` then
    walks the ``from_node`` chain to assign integer handles.
    """
    if not _has_meta_key(ep.graph_module, "from_node"):
        # Re-trace to populate from_node; produces a new ep, source stays
        # untouched. After re-tracing, any prior debug_handle is lost from the
        # newly-created nodes, so force a re-derive.
        ep = ep.run_decompositions({})

    if not _has_meta_key(ep.graph_module, "debug_handle"):
        DebugHandleGeneratorPass()(ep.graph_module)

    return ep


def _build_backend_overlay(rec) -> Optional[GraphExtension]:  # noqa: C901
    """Return a ``backend`` GraphExtension over the edge graph, or ``None``.

    Sources the backend name from the *outer* ``_delegate_map`` instruction
    table (``{method: {instr_id: {"name": backend, ...}}}``) crossed with the
    ``_debug_handle_map`` instruction-to-handle mapping. Some backends leave
    the inner ``entry["delegate_map"]`` empty, so relying on it alone tags
    zero nodes.
    """
    dhm = rec._debug_handle_map or {}
    dm = rec._delegate_map or {}
    if not dhm or not dm:
        return None

    edge_gm = rec.edge_dialect_program.graph_module
    h_to_nodes: dict[int, list[str]] = {}
    for n in edge_gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        h = n.meta.get("debug_handle")
        if h in (None, 0):
            continue
        for hi in h if isinstance(h, (list, tuple)) else [h]:
            try:
                h_to_nodes.setdefault(int(hi), []).append(n.name)
            except (TypeError, ValueError):
                continue

    overlay: dict[str, str] = {}
    saw_backend = False
    for method, method_dhm in dhm.items():
        dm_method = dm.get(method) or {}
        if not isinstance(dm_method, dict):
            dm_method = {}
        for instr_id, handles in method_dhm.items():
            entry = dm_method.get(instr_id) or dm_method.get(str(instr_id))
            backend = (
                entry.get("name") if isinstance(entry, dict) else None
            ) or "portable"
            if backend != "portable":
                saw_backend = True
            if isinstance(handles, int):
                handles = [handles]
            for h in handles or []:
                try:
                    hi = int(h)
                except (TypeError, ValueError):
                    continue
                for name in h_to_nodes.get(hi, []):
                    overlay[name] = backend

    if not overlay or not saw_backend:
        return None

    ext = GraphExtension(id="backend", name="Backend Assignment")
    for name, backend in overlay.items():
        ext.add_node_data(name, {"backend": backend})
    ext.set_color_rule(CategoricalColorRule(attribute="backend"))
    return ext


def _prettify_extra_key(key: str) -> str:
    """``"edge_after_transform/forward"`` -> ``"Edge After Transform"``."""
    stem = key.split("/", 1)[0]
    return stem.replace("_", " ").title()


def build_compare_from_etrecord(
    etrecord: Union[str, ETRecord],
    *,
    title: str = "ETRecord Compare",
    enrich_missing_meta: bool = True,
    include_backend_overlay: bool = True,
) -> FXGraphCompareExporter:
    """Turn an ``.etrecord`` bundle into a ready-to-export compare exporter.

    Included panes (each gated on availability in the bundle):
      * ``"Aten"`` from ``rec.exported_program`` (enriched if needed).
      * One pane per ``rec.graph_map[<name>/<method>]`` entry — including
        ``"edge_after_transform"`` which the exir pipeline writes when
        ``to_edge_transform_and_lower(..., transform_passes=...)`` is used.
      * ``"Edge dialect"`` from ``rec.edge_dialect_program``, with a
        ``backend`` overlay when ``_delegate_map`` names a backend.

    Args:
        etrecord: path to a ``.etrecord`` file, or an in-memory ``ETRecord``.
        title: page title.
        enrich_missing_meta: when ``True`` (default), re-run
            ``run_decompositions({})`` and ``DebugHandleGeneratorPass`` on
            any stored program that lacks ``from_node`` or ``debug_handle``.
        include_backend_overlay: when ``True`` (default), attach a ``backend``
            ``GraphExtension`` to the edge pane if the bundle names any.
    """
    rec = etrecord if isinstance(etrecord, ETRecord) else parse_etrecord(etrecord)

    viewers: "OrderedDict[str, FXGraphExporter]" = OrderedDict()

    if rec.exported_program is not None:
        aten_ep = rec.exported_program
        if enrich_missing_meta:
            aten_ep = _enrich_missing_meta(aten_ep)
        viewers["Aten"] = FXGraphExporter(aten_ep.graph_module)

    for key in sorted((rec.graph_map or {}).keys()):
        prog = rec.graph_map[key]
        if enrich_missing_meta:
            prog = _enrich_missing_meta(prog)
        viewers[_prettify_extra_key(key)] = FXGraphExporter(prog.graph_module)

    if rec.edge_dialect_program is not None:
        edge_prog = rec.edge_dialect_program
        if enrich_missing_meta:
            edge_prog = _enrich_missing_meta(edge_prog)
        edge_exp = FXGraphExporter(edge_prog.graph_module)
        if include_backend_overlay:
            ext = _build_backend_overlay(rec)
            if ext is not None:
                edge_exp.add_extension(ext)
        viewers["Edge dialect"] = edge_exp

    active_extensions = ["backend"] if include_backend_overlay else []
    color_by = "backend" if include_backend_overlay else None

    return FXGraphCompareExporter(
        viewers,
        title=title,
        sync_mode="auto",
        active_extensions=active_extensions,
        color_by=color_by,
    )


def export_etrecord_to_html(
    etrecord: Union[str, ETRecord],
    output_html: str,
    *,
    title: str = "ETRecord Compare",
    enrich_missing_meta: bool = True,
    include_backend_overlay: bool = True,
) -> None:
    """One-liner: ``.etrecord`` -> interactive compare HTML."""
    build_compare_from_etrecord(
        etrecord,
        title=title,
        enrich_missing_meta=enrich_missing_meta,
        include_backend_overlay=include_backend_overlay,
    ).export_html(output_html)
