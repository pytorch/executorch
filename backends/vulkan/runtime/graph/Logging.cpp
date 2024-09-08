/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <set>

namespace vkcompute {

void ComputeGraph::print_readable() {
  std::set<ValueRef> input_set;
  for (const IOValueRef& io_val : inputs()) {
    input_set.insert(io_val.value);
  }

  std::set<ValueRef> output_set;
  for (const IOValueRef& io_val : outputs()) {
    output_set.insert(io_val.value);
  }

  std::set<ValueRef> prepack_set;
  for (const std::unique_ptr<PrepackNode>& node : prepack_nodes()) {
    prepack_set.insert(node->tref_);
    prepack_set.insert(node->packed_);
  }

  std::map<ValueRef, size_t> value_ref_to_shared_object_idx;

  std::cout << "====================" << std::left << std::setfill('=')
            << std::setw(40) << " Shared Object List " << std::right
            << std::setfill(' ') << std::endl;

  std::cout << std::setw(6) << "idx" << std::setw(20) << "sizes"
            << std::setw(24) << "users" << std::endl;

  size_t so_idx = 0;
  for (const SharedObject& shared_object : shared_objects_) {
    std::cout << std::setw(6) << so_idx;
    {
      std::stringstream ss;
      ss << shared_object.aggregate_memory_requirements.size;
      std::cout << std::setw(20) << ss.str();
    }

    {
      std::stringstream ss;
      ss << shared_object.users;
      std::cout << std::setw(24) << ss.str();
    }
    std::cout << std::endl;

    for (const ValueRef& user : shared_object.users) {
      value_ref_to_shared_object_idx[user] = so_idx;
    }

    so_idx++;
  }

  std::cout << "====================" << std::left << std::setfill('=')
            << std::setw(40) << " Value List " << std::right
            << std::setfill(' ') << std::endl;

  std::cout << std::setw(6) << "idx" << std::setw(10) << "type" << std::setw(20)
            << "sizes" << std::setw(10) << "node_type" << std::setw(15)
            << "storage_bytes" << std::setw(10) << "so_idx" << std::endl;

  size_t value_idx = 0;
  for (Value& val : values_) {
    std::cout << std::setw(6) << value_idx << std::setw(10) << val.type();

    // sizes
    std::cout << std::setw(20);
    if (val.isTensor()) {
      const api::vTensor& v_tensor = val.toTensor();
      std::stringstream ss;
      ss << v_tensor.sizes();
      std::cout << ss.str();
    } else if (val.isTensorRef()) {
      const TensorRef tensor_ref = val.toTensorRef();
      std::stringstream ss;
      ss << tensor_ref.sizes;
      std::cout << ss.str();
    } else {
      std::cout << "";
    }

    // Node type
    std::cout << std::setw(10);
    {
      if (input_set.count(value_idx) > 0) {
        std::cout << "INPUT";
      } else if (output_set.count(value_idx) > 0) {
        std::cout << "OUTPUT";
      } else if (prepack_set.count(value_idx) > 0) {
        std::cout << "PREPACK";
      } else {
        std::cout << "";
      }
    }

    // Actual storage bytes used
    std::cout << std::setw(15);
    if (val.isTensor()) {
      const api::vTensor& v_tensor = val.toTensor();
      auto memory_reqs = v_tensor.get_memory_requirements();
      std::cout << memory_reqs.size;
    } else {
      std::cout << "";
    }

    std::cout << std::setw(10);
    if (value_ref_to_shared_object_idx.count(value_idx) > 0) {
      size_t shared_obj_idx = value_ref_to_shared_object_idx.at(value_idx);
      std::cout << shared_obj_idx;
    } else {
      std::cout << "";
    }

    std::cout << std::endl;
    value_idx++;
  }

  std::cout << "====================" << std::left << std::setfill('=')
            << std::setw(40) << " Prepack Node List " << std::right
            << std::setfill(' ') << std::endl;
  std::cout << std::setw(6) << "idx" << std::setw(32) << "shader_name"
            << std::setw(8) << "tref" << std::setw(8) << "packed" << std::endl;

  size_t prepack_node_idx = 0;
  for (const std::unique_ptr<PrepackNode>& node : prepack_nodes()) {
    std::cout << std::setw(6) << prepack_node_idx << std::setw(32)
              << node->shader_.kernel_name << std::setw(8) << node->tref_
              << std::setw(8) << node->packed_ << std::endl;

    prepack_node_idx++;
  }

  std::cout << "====================" << std::left << std::setfill('=')
            << std::setw(40) << " Execute Node List " << std::right
            << std::setfill(' ') << std::endl;

  std::cout << std::setw(6) << "idx" << std::setw(32) << "shader_name"
            << std::setw(24) << "READ_arg" << std::setw(24) << "WRITE_arg"
            << std::endl;

  size_t node_idx = 0;
  for (const std::unique_ptr<ExecuteNode>& node : execute_nodes()) {
    std::cout << std::setw(6) << node_idx;
    std::cout << std::setw(32) << node->shader_.kernel_name;

    std::stringstream read_s;
    for (const ArgGroup& arg_group : node->args_) {
      if (arg_group.access != vkapi::MemoryAccessType::READ) {
        continue;
      }
      read_s << arg_group.refs;
    }
    std::cout << std::setw(24) << read_s.str();

    std::stringstream write_s;
    for (const ArgGroup& arg_group : node->args_) {
      if (arg_group.access != vkapi::MemoryAccessType::WRITE) {
        continue;
      }
      write_s << arg_group.refs;
    }
    std::cout << std::setw(24) << write_s.str();

    std::cout << std::endl;

    node_idx++;
  }
}

} // namespace vkcompute
