/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run static mimi decoder with Qualcomm AI Engine
 * Direct.
 *
 */

#include <executorch/examples/qualcomm/oss_scripts/moshi/runner/runner.h>
#include <executorch/runtime/platform/runtime.h>
#include <gflags/gflags.h>

DEFINE_string(
    model_path,
    "mimi_decoder_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");
DEFINE_string(
    input_list_path,
    "input_list.txt",
    "Input list storing file name of encoded results.");

using executorch::runtime::Error;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  example::Runner runner(FLAGS_model_path, FLAGS_output_folder_path);

  ET_CHECK_MSG(
      runner.generate(FLAGS_input_list_path) == Error::Ok,
      "Runner failed to generate");

  return 0;
}
