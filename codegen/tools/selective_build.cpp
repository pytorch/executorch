/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/assert.h>
#include <executorch/schema/program_generated.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef ET_BUNDLE_IO
#include <executorch/devtools/bundled_program/bundled_program.h>
#include <stdexcept>
#endif

namespace py = pybind11;

namespace torch {
namespace executor {

namespace {

// Metadata for kernel call io variables.
// dtype and dim_order will exist only if corresponding variable is Tensor.
struct IOMetaData {
  int kernel_type;
  int dtype;
  std::vector<unsigned int> dim_order;

  // Create tensor metadata. It records tensor's dtype and dim order.
  explicit IOMetaData(const executorch_flatbuffer::Tensor* t)
      : kernel_type(
            static_cast<int>(executorch_flatbuffer::KernelTypes::Tensor)),
        dtype(static_cast<int>(t->scalar_type())) {
    for (size_t i = 0; i < t->dim_order()->size(); i++) {
      dim_order.push_back(static_cast<unsigned int>(t->dim_order()->Get(i)));
    }
  }

  // Create metadata for non-tensor variable.
  explicit IOMetaData(executorch_flatbuffer::KernelTypes type)
      : kernel_type(static_cast<int>(type)) {
    ET_CHECK(
        type != executorch_flatbuffer::KernelTypes::Tensor &&
        type != executorch_flatbuffer::KernelTypes::TensorList &&
        type != executorch_flatbuffer::KernelTypes::OptionalTensorList);
  }
};

struct KernelIOMetaDataComparsion {
  bool operator()(
      const std::vector<IOMetaData>& lhs,
      const std::vector<IOMetaData>& rhs) const {
    if (lhs.size() != rhs.size()) {
      return lhs.size() < rhs.size();
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i].kernel_type != rhs[i].kernel_type) {
        return lhs[i].kernel_type < rhs[i].kernel_type;
      }
      if (lhs[i].kernel_type !=
          static_cast<int>(executorch_flatbuffer::KernelTypes::Tensor)) {
        continue;
      }
      if (lhs[i].dtype != rhs[i].dtype) {
        return lhs[i].dtype < rhs[i].dtype;
      }
      if (lhs[i].dim_order != rhs[i].dim_order) {
        return lhs[i].dim_order < rhs[i].dim_order;
      }
    }
    return false;
  }
};

using KernelIOMetadata = std::vector<IOMetaData>;

using OpIOMetaData = std::set<KernelIOMetadata, KernelIOMetaDataComparsion>;

std::vector<std::string> get_operators_from_execution_plan(
    const executorch_flatbuffer::ExecutionPlan& plan) {
  std::vector<std::string> op_names;
  for (const executorch_flatbuffer::Operator* op : *plan.operators()) {
    if (op->overload()->str().empty()) {
      op_names.push_back(op->name()->str());
    } else {
      op_names.push_back(op->name()->str() + "." + op->overload()->str());
    }
  }
  return op_names;
}

std::map<std::string, OpIOMetaData>
get_kernel_tensor_metadatas_from_execution_plan(
    const executorch_flatbuffer::ExecutionPlan* plan) {
  std::map<std::string, OpIOMetaData> op_io_metadata;
  for (const executorch_flatbuffer::Chain* chain : *plan->chains()) {
    for (const executorch_flatbuffer::Instruction* inst :
         *chain->instructions()) {
      if (inst->instr_args_type() ==
          executorch_flatbuffer::InstructionArguments::KernelCall) {
        const executorch_flatbuffer::KernelCall* kernel_call =
            inst->instr_args_as_KernelCall();
        const executorch_flatbuffer::Operator* op =
            plan->operators()->Get(kernel_call->op_index());
        std::string op_overload_name = op->name()->str();
        if (op->overload()->size()) {
          op_overload_name += "." + op->overload()->str();
        }

        // create an empty entry if current kernel is not in the map.
        if (op_io_metadata.count(op_overload_name) == 0) {
          op_io_metadata.insert(
              std::make_pair(op_overload_name, OpIOMetaData()));
        }

        // go through IOs of this operator and collect tensor metadatas.
        KernelIOMetadata kernel_io_metadata;
        for (int arg_id : *kernel_call->args()) {
          const executorch_flatbuffer::EValue* arg =
              plan->values()->Get(arg_id);
          if (arg->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
            kernel_io_metadata.push_back(IOMetaData(arg->val_as_Tensor()));
          } else if (
              arg->val_type() ==
              executorch_flatbuffer::KernelTypes::TensorList) {
            if (arg->val_as_TensorList()->items()->size() == 0) {
              // treat empty tensor list as null type since we can not get
              // metadata from it.
              kernel_io_metadata.push_back(
                  IOMetaData(executorch_flatbuffer::KernelTypes::Null));
            } else {
              // all eles in TensorList are tensor and share same tensor
              // metadata. use the metadata of first element as the metadata for
              // whole list.
              const executorch_flatbuffer::Tensor* tensor_arg =
                  plan->values()
                      ->Get(arg->val_as_TensorList()->items()->Get(0))
                      ->val_as_Tensor();
              kernel_io_metadata.push_back(IOMetaData(tensor_arg));
            }
          } else if (
              arg->val_type() ==
              executorch_flatbuffer::KernelTypes::OptionalTensorList) {
            // all eles in OptionalTensorList are either tensor or null, and all
            // tensors share same metadata. Use the metadata of first tensor
            // element as the metadata for whole list. If no tensor exists (e.g.
            // each element is None), treat the whole list as a single null
            // element.
            const executorch_flatbuffer::OptionalTensorList* opt_tensor_list =
                arg->val_as_OptionalTensorList();

            // Find one non-null tensor
            bool found_tensor_element = false;
            for (size_t i = 0; i < opt_tensor_list->items()->size(); i++) {
              // We now adopt both index == -1 and actually serialize a null
              // type EValue to represent a null data.
              if (opt_tensor_list->items()->Get(i) != -1 &&
                  plan->values()
                          ->Get(opt_tensor_list->items()->Get(i))
                          ->val_type() ==
                      executorch_flatbuffer::KernelTypes::Tensor) {
                const executorch_flatbuffer::Tensor* tensor_arg =
                    plan->values()
                        ->Get(opt_tensor_list->items()->Get(i))
                        ->val_as_Tensor();
                kernel_io_metadata.push_back(IOMetaData(tensor_arg));
                found_tensor_element = true;
                break;
              }
            }
            if (!found_tensor_element) {
              kernel_io_metadata.push_back(
                  IOMetaData(executorch_flatbuffer::KernelTypes::Null));
            }
          } else {
            kernel_io_metadata.push_back(IOMetaData(arg->val_type()));
          }
        }
        op_io_metadata[op_overload_name].insert(kernel_io_metadata);
      }
    }
  }
  return op_io_metadata;
}
} // namespace

const executorch_flatbuffer::Program* _get_program_from_buffer(
    const py::bytes& buffer) {
  // Access the Python bytes without copying and get raw pointer/size.
  const std::string_view sv = buffer.cast<std::string_view>();
#ifdef ET_BUNDLE_IO
  void* buf_ptr = const_cast<void*>(static_cast<const void*>(sv.data()));
  const size_t buf_len = sv.size();

  // If this is a bundled program, extract the inner ExecuTorch program bytes.
  if (executorch::bundled_program::is_bundled_program(buf_ptr, buf_len)) {
    const void* program_data = nullptr;
    size_t program_size = 0;

    const auto status = executorch::bundled_program::get_program_data(
        buf_ptr, // serialized BundledProgram start
        buf_len, // total size of the BundledProgram blob
        &program_data, // [out] pointer to inner .pte bytes
        &program_size // [out] size of inner .pte bytes
    );

    if (status != ::executorch::runtime::Error::Ok || program_data == nullptr ||
        program_size == 0) {
      throw std::runtime_error(
          "bundled_program::get_program_data() failed or returned empty data");
    }

    // program_data points directly at the flatbuffer-encoded Program region.
    return executorch_flatbuffer::GetProgram(
        reinterpret_cast<const uint8_t*>(program_data));
  }
#endif
  // Otherwise treat the buffer as a raw .pte (flatbuffer Program with optional
  // extended header).
  return executorch_flatbuffer::GetProgram(
      reinterpret_cast<const uint8_t*>(sv.data()));
}

py::list _get_program_operators(const executorch_flatbuffer::Program* program) {
  const auto& plans = *program->execution_plan();
  std::vector<std::string> op_names;
  for (const auto& plan : plans) {
    auto plan_ops = get_operators_from_execution_plan(*plan);
    if (!plan_ops.empty()) {
      op_names.insert(op_names.end(), plan_ops.begin(), plan_ops.end());
    }
  }
  return py::cast(op_names);
}

// expose IO metadatas for all operators in given program
py::dict _get_io_metadata_for_program_operators(
    const executorch_flatbuffer::Program* program) {
  const auto& plans = *program->execution_plan();
  std::map<std::string, OpIOMetaData> program_op_io_metadata;

  // aggregrate op metadata from different execution plan.
  for (const executorch_flatbuffer::ExecutionPlan* plan : plans) {
    std::map<std::string, OpIOMetaData> plan_op_io_metadata =
        get_kernel_tensor_metadatas_from_execution_plan(plan);

    for (const auto& op_io_metadata : plan_op_io_metadata) {
      std::string op_name = op_io_metadata.first;
      if (program_op_io_metadata.count(op_name) == 0) {
        program_op_io_metadata.insert(std::make_pair(op_name, OpIOMetaData()));
      }
      program_op_io_metadata[op_name].insert(
          plan_op_io_metadata[op_name].begin(),
          plan_op_io_metadata[op_name].end());
    }
  }

  // convert program_op_io_metadata to py data structure.
  py::dict py_program_op_io_metadata;
  for (const auto& op_io_meta : program_op_io_metadata) {
    py::set py_op_io_meta;
    for (const auto& io_metas : op_io_meta.second) {
      py::list py_io_metadatas;
      for (const auto& io_metadata : io_metas) {
        py_io_metadatas.append(io_metadata);
      }
      py_op_io_meta.add(py::tuple(py_io_metadatas));
    }
    py_program_op_io_metadata[op_io_meta.first.data()] = py_op_io_meta;
  }

  return py_program_op_io_metadata;
}

PYBIND11_MODULE(EXECUTORCH_PYTHON_MODULE_NAME, m) {
  py::class_<executorch_flatbuffer::Program>(m, "_Program");

  m.def(
      "_get_program_from_buffer",
      &_get_program_from_buffer,
      py::return_value_policy::reference);

  m.def(
      "_get_program_operators",
      &_get_program_operators,
      py::return_value_policy::copy);

  m.def(
      "_get_io_metadata_for_program_operators",
      &_get_io_metadata_for_program_operators,
      py::return_value_policy::copy);

  py::class_<IOMetaData>(m, "_IOMetaData")
      .def_readwrite("kernel_type", &IOMetaData::kernel_type)
      .def_readwrite("dtype", &IOMetaData::dtype)
      .def_readwrite("dim_order", &IOMetaData::dim_order);
}

} // namespace executor
} // namespace torch
