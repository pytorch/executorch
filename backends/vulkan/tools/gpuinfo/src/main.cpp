/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "app.h"
#include "architecture.h"
#include "buffers.h"
#include "textures.h"

using namespace vkapi;

int main(int argc, const char** argv) {
  gpuinfo::App app;

  std::string file_path = "config.json";
  if (argc > 1) {
    file_path = argv[1];
  };
  app.load_config(file_path);

  // Architecture
  gpuinfo::reg_count(app);
  gpuinfo::warp_size(app);

  // Buffers
  gpuinfo::buf_cacheline_size(app);
  gpuinfo::buf_bandwidth(app);
  gpuinfo::ubo_bandwidth(app);
  gpuinfo::shared_mem_bandwidth(app);

  // Textures
  gpuinfo::tex_bandwidth(app);
  gpuinfo::tex_cacheline_concurr(app);

  return 0;
}
