/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <fstream>
#include <vector>

#include "dsp_capabilities_utils.h"
#include "qnn_executorch.h"
#include "remote.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    output_folder_path,
    ".",
    "Executorch inference data output path.");
DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");
DEFINE_bool(
    shared_buffer,
    false,
    "UNSUPPORTED: shared buffer is not yet supported in direct model.");
DEFINE_uint32(method_index, 0, "Index of methods to be specified.");
DEFINE_string(
    etdump_path,
    "etdump.etdp",
    "If etdump generation is enabled an etdump will be written out to this path");

DEFINE_bool(
    dump_intermediate_outputs,
    false,
    "Dump intermediate outputs to etdump file.");

DEFINE_string(
    debug_output_path,
    "debug_output.bin",
    "Path to dump debug outputs to.");

DEFINE_int32(
    debug_buffer_size,
    100000000, // 100MB
    "Size of the debug buffer in bytes to allocate for intermediate outputs and program outputs logging.");

DEFINE_int32(
    domain_id,
    3, // CDSP=3
    "The domain fastrpc communicates with.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // For domain_id 0 (ADSP), use signed_pd.
  // For other domain (CDSP for now), use unsigned_pd.
  bool is_unsignedpd_requested = (FLAGS_domain_id != 0);
  std::string domain_uri(qnn_executorch_URI);

  domain* my_domain = NULL;
  char* uri = NULL;
  remote_handle64 handle = -1;
  int remote_status = AEE_SUCCESS;

  my_domain = get_domain(FLAGS_domain_id);
  ET_CHECK_MSG(my_domain != NULL, "Unable to find domain %d", FLAGS_domain_id);
  uri = my_domain->uri;
  domain_uri.append(uri);
  ET_LOG(Info, "Domain URI for direct mode: %s", domain_uri.c_str());

  struct remote_rpc_control_unsigned_module data;
  data.domain = FLAGS_domain_id;
  data.enable = is_unsignedpd_requested;

  remote_status = remote_session_control(
      DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data));
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "remote_session_control failed: 0x%x",
      remote_status);

  remote_status = qnn_executorch_open(domain_uri.data(), &handle);
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "qnn_executorch_open failed: 0x%x",
      remote_status);

  // load model
  const char* model_path = FLAGS_model_path.c_str();
  remote_status = qnn_executorch_load(handle, model_path, FLAGS_method_index);
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "qnn_executorch_load failed: 0x%x",
      remote_status);

  if (FLAGS_dump_intermediate_outputs) {
    remote_status = qnn_executorch_enable_intermediate_tensor_dump(
        handle, model_path, FLAGS_debug_buffer_size);
    ET_CHECK_MSG(
        remote_status == AEE_SUCCESS,
        "qnn_executorch_enable_intermediate_tensor_dump failed: 0x%x",
        remote_status);
  }

  int num_inferences_performed = 0;
  double total_execute_interval = 0;
  double total_read_file_interval = 0;
  double total_save_file_interval = 0;

  auto execute_all_start = std::chrono::high_resolution_clock::now();
  remote_status = qnn_executorch_execute_all(
      handle,
      model_path,
      FLAGS_input_list_path.c_str(),
      FLAGS_output_folder_path.c_str(),
      &num_inferences_performed,
      &total_execute_interval,
      &total_read_file_interval,
      &total_save_file_interval);
  auto execute_all_end = std::chrono::high_resolution_clock::now();
  double execute_all_interval =
      std::chrono::duration_cast<std::chrono::microseconds>(
          execute_all_end - execute_all_start)
          .count() /
      1000.0;
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "qnn_executorch_execute_all failed: 0x%x",
      remote_status);

  ET_LOG(
      Info,
      "qnn_executorch_execute_all took a total of %f ms, performing %d inference, avg %f ms.",
      execute_all_interval,
      num_inferences_performed,
      execute_all_interval / num_inferences_performed);
  ET_LOG(
      Info,
      "Method execution took a total of %f ms, performing %d inference, avg %f ms.",
      total_execute_interval,
      num_inferences_performed,
      total_execute_interval / num_inferences_performed);
  ET_LOG(
      Info,
      "Reading data from file to tensor took a total of %f ms, performing %d inference, avg %f ms.",
      total_read_file_interval,
      num_inferences_performed,
      total_read_file_interval / num_inferences_performed);
  ET_LOG(
      Info,
      "Saving output from tensor to file took a total of %f ms, performing %d inference, avg %f ms.",
      total_save_file_interval,
      num_inferences_performed,
      total_save_file_interval / num_inferences_performed);

  remote_status =
      qnn_executorch_dump_etdp(handle, model_path, FLAGS_etdump_path.c_str());
  if (remote_status != AEE_SUCCESS) {
    ET_LOG(Info, "Failed to generate etdp file.");
  }

  if (FLAGS_dump_intermediate_outputs) {
    remote_status = qnn_executorch_dump_intermediate_tensor(
        handle, model_path, FLAGS_debug_output_path.c_str());
    ET_CHECK_MSG(
        remote_status == AEE_SUCCESS,
        "qnn_executorch_dump_intermediate_tensor failed: 0x%x",
        remote_status);
  }

  // unload model
  remote_status = qnn_executorch_unload(handle, model_path);
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "qnn_executorch_unload failed: 0x%x",
      remote_status);
  // tear down
  remote_status = qnn_executorch_close(handle);
  ET_CHECK_MSG(
      remote_status == AEE_SUCCESS,
      "qnn_executorch_close failed: 0x%x",
      remote_status);
  return 0;
}
