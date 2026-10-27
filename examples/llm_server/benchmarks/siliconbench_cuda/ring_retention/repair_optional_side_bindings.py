# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Repair ABI4's optional seq_cols binding in an isolated native source copy."""
from pathlib import Path


def replace_once(text, before, after):
    assert text.count(before) == 1, (before, text.count(before))
    return text.replace(before, after)


def patch(source):
    source = Path(source)
    path = source / "backends/cuda/runtime/cuda_kv_pool.h"
    text = path.read_text()
    text = replace_once(text, "    std::vector<int64_t> sizes;\n  };",
        "    std::vector<int64_t> sizes;\n    bool required = true;\n  };")
    path.write_text(text)
    path = source / "backends/cuda/runtime/cuda_kv_pool.cpp"
    text = path.read_text()
    text = replace_once(text, "    if (discovered_fqns_.count(spec.fqn) == 0) {",
        "    if (spec.required && discovered_fqns_.count(spec.fqn) == 0) {")
    text = replace_once(text, "  size_t found_side = 0;", "  size_t found_side = 0;\n  size_t required_side = 0;\n  size_t found_required_side = 0;")
    text = replace_once(text, "    const SideBuffer& spec = side_specs_[index];\n    const auto it = compiled.find(spec.fqn);",
        "    const SideBuffer& spec = side_specs_[index];\n    required_side += spec.required;\n    const auto it = compiled.find(spec.fqn);")
    text = replace_once(text, "      ++found_side;", "      ++found_side;\n      found_required_side += spec.required;")
    text = replace_once(text, "      found_side == (found_layers == 0 ? 0 : side_specs_.size()),",
        "      found_layers == 0 ? found_side == 0 : found_required_side == required_side,")
    text = replace_once(text, '      "offgraph_kv: program carries %zu of %zu side buffers",\n      found_side,\n      side_specs_.size());',
        '      "offgraph_kv: program carries %zu of %zu required side buffers",\n      found_required_side,\n      required_side);')
    path.write_text(text)
    path = source / "backends/cuda/runtime/cuda_kv_cache.cpp"
    text = replace_once(path.read_text(),
        "        {kSeqColsFqn, slimc10::ScalarType::Long, {*cfg.max_write}},",
        "        {kSeqColsFqn, slimc10::ScalarType::Long, {*cfg.max_write}, !cfg.split_ring_history},")
    path.write_text(text)
