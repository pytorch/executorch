/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <espnn/xts3_graph.h>

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::FILE* file = std::fopen(argv[i], "rb");
    if (file == nullptr) {
      std::printf("%s missing\n", argv[i]);
      return 2;
    }
    std::vector<uint8_t> data;
    uint8_t chunk[4096];
    size_t read = 0;
    while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
      data.insert(data.end(), chunk, chunk + read);
    }
    std::fclose(file);
    executorch::backends::espressif::GraphView graph{};
    const bool accepted = executorch::backends::espressif::parse_graph(
        data.data(), data.size(), graph);
    std::printf("VERDICT %s %s\n", argv[i], accepted ? "accept" : "reject");
  }
  return 0;
}
