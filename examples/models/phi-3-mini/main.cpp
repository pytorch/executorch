/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/phi-3-mini/runner.h>

int main(int32_t argc, char** argv) {
  const char* model_path =
      "/home/lunwenh/executorch/examples/models/phi-3-mini/example.pte";

  ::torch::executor::Runner runner(model_path);

  runner.generate();

  return 0;
}
