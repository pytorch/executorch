// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

namespace ptn {

class Program;

// Render `program` as Graphviz DOT text: methods as clusters, def-use as
// labeled edges, plus tensor metadata, constants, alias / mutation facts and
// HOP subgraphs. Pure string builder -- the caller writes or renders it (e.g.
// `dot -Tpng`).
//
// A free function in its own target rather than a Program member, so a
// production link that only reads programs does not pull in the renderer.
std::string to_dot(const Program& program);

} // namespace ptn
