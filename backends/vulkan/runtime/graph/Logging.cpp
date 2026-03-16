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

std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& sizes) {
  if (sizes.size() == 0) {
    os << "[]";
    return os;
  }
  os << "[";
  for (int i = 0; i < sizes.size() - 1; ++i) {
    os << sizes.at(i) << ", ";
  }
  os << sizes.at(sizes.size() - 1);
  os << "]";
  return os;
}

std::string make_arg_json(ComputeGraph* const compute_graph, ValueRef arg) {
  std::stringstream ss;
  ss << "{\"type\": \"" << compute_graph->get_val_type(arg) << "\", ";
  ss << "\"value_ref\": " << arg;
  if (compute_graph->val_is_tensor(arg)) {
    ss << ", \"dtype\": \"";
    ss << compute_graph->dtype_of(arg) << "\"";
    ss << ", \"sizes\": ";
    ss << compute_graph->sizes_of(arg);
    ss << ", \"storage\": \"";
    ss << compute_graph->storage_type_of(arg) << "\"";
    ss << ", \"packed_dim\": ";
    ss << compute_graph->packed_dim_of(arg);
  } else if (compute_graph->val_is_tref(arg)) {
    ss << ", \"sizes\": ";
    ss << compute_graph->sizes_of(arg);
    ss << ", \"dtype\": \"";
    ss << compute_graph->dtype_of(arg) << "\"";
  } else if (compute_graph->val_is_value_list(arg)) {
    ValueListPtr val_list = compute_graph->get_value_list(arg);
    ss << ", \"values\": [";
    for (size_t i = 0; i < val_list->size(); ++i) {
      ss << val_list->at(i);
      if (i + 1 < val_list->size()) {
        ss << ", ";
      }
    }
    ss << "]";
  } else if (compute_graph->val_is_int_list(arg)) {
    ss << ", \"values\": ";
    ss << *compute_graph->get_int_list(arg);
  } else if (compute_graph->val_is_int(arg)) {
    ss << ", \"value\": ";
    ss << compute_graph->get_int(arg);
  } else if (compute_graph->val_is_double(arg)) {
    ss << ", \"value\": ";
    ss << compute_graph->get_double(arg);
  } else if (compute_graph->val_is_bool(arg)) {
    ss << ", \"value\": ";
    ss << compute_graph->get_bool(arg);
  } else if (compute_graph->val_is_symint(arg)) {
    ss << ", \"value\": ";
    ss << compute_graph->read_symint(arg);
  }
  ss << "}";

  return ss.str();
}

std::string make_operator_json(
    ComputeGraph* const compute_graph,
    std::string& op_name,
    std::vector<ValueRef>& args) {
  std::stringstream ss;
  ss << "\"name\": \"" << op_name << "\", \"args\": [";
  for (size_t i = 0; i < args.size(); ++i) {
    ss << make_arg_json(compute_graph, args[i]);
    if (i + 1 < args.size()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

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
      const TensorRef& tensor_ref = val.toTensorRef();
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
    std::cout << std::setw(32) << node->name();

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
