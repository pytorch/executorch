/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/NativeBackendInternal.h>

#include <new>
#include <optional>

namespace executorch {
namespace backends {
namespace native {

// Initialize one tensor EValue from the flatbuffer Tensor metadata.
Error initialize_tensor_evalue(DelegateInstance* d, uint32_t value_id) {
  const auto& graph = *d->graph;
  auto sizes = graph.tensor_sizes(value_id);
  auto dim_order_in = graph.tensor_dim_order(value_id);
  size_t dim = sizes.size();

  DelegateInstance::TensorMeta meta;
  meta.sizes.reset(new SizesType[dim]);
  meta.dim_order.reset(new DimOrderType[dim]);
  meta.strides.reset(new StridesType[dim]);

  for (size_t i = 0; i < dim; ++i) {
    meta.sizes[i] = sizes[i];
  }
  if (dim_order_in.size() == dim) {
    for (size_t i = 0; i < dim; ++i) {
      meta.dim_order[i] = dim_order_in[i];
    }
  } else {
    for (size_t i = 0; i < dim; ++i) {
      meta.dim_order[i] = static_cast<DimOrderType>(i);
    }
  }
  auto status = ::executorch::runtime::dim_order_to_stride(
      meta.sizes.get(), meta.dim_order.get(), dim, meta.strides.get());
  if (status != Error::Ok)
    return status;

  ScalarType dtype = graph.tensor_dtype(value_id);
  meta.impl.reset(new TensorImpl(
      dtype,
      static_cast<ssize_t>(dim),
      meta.sizes.get(),
      /*data=*/nullptr,
      meta.dim_order.get(),
      meta.strides.get(),
      ::executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND));

  d->values[value_id] = EValue(Tensor(meta.impl.get()));
  d->tensor_metas.push_back(std::move(meta));
  return Error::Ok;
}

Error initialize_values(
    DelegateInstance* d,
    ::executorch::runtime::MemoryAllocator* runtime_alloc) {
  using ::executorch::runtime::BoxedEvalueList;

  size_t n = d->graph->num_values();
  if (n == 0)
    return Error::Ok;

  if (d->values.size() < n) {
    return Error::InvalidState;
  }

  for (uint32_t i = 0; i < n; ++i) {
    switch (d->graph->value_type(i)) {
      case ValueType::None:
        d->values[i] = EValue();
        break;
      case ValueType::Int:
        d->values[i] = EValue(d->graph->int_value(i));
        break;
      case ValueType::Double:
        d->values[i] = EValue(d->graph->double_value(i));
        break;
      case ValueType::Bool:
        d->values[i] = EValue(d->graph->bool_value(i));
        break;
      case ValueType::Tensor:
        if (auto e = initialize_tensor_evalue(d, i); e != Error::Ok) {
          return e;
        }
        break;
      case ValueType::IntList: {
        auto items = d->graph->int_list_member_ids(i);
        size_t cnt = items.size();
        EValue** evalp_list = runtime_alloc->allocateList<EValue*>(cnt);
        int64_t* int_list = runtime_alloc->allocateList<int64_t>(cnt);
        if (!evalp_list || !int_list) {
          return Error::MemoryAllocationFailed;
        }
        for (size_t j = 0; j < cnt; ++j) {
          int64_t vidx = items[j];
          if (vidx < 0 || static_cast<size_t>(vidx) >= n) {
            return Error::InvalidProgram;
          }
          evalp_list[j] = &d->values[static_cast<size_t>(vidx)];
        }
        auto* boxed_mem =
            runtime_alloc->allocateInstance<BoxedEvalueList<int64_t>>();
        if (!boxed_mem)
          return Error::MemoryAllocationFailed;
        auto* boxed =
            new (boxed_mem) BoxedEvalueList<int64_t>(evalp_list, int_list, cnt);
        d->values[i] = EValue(boxed);
      } break;
      case ValueType::TensorList:
        d->values[i] = EValue();
        break;
      case ValueType::OptionalTensorList: {
        auto items = d->graph->tensor_list_member_ids(i);
        size_t cnt = items.size();
        EValue** evalp_list = runtime_alloc->allocateList<EValue*>(cnt);
        std::optional<::executorch::aten::Tensor>* opt_list =
            runtime_alloc
                ->allocateList<std::optional<::executorch::aten::Tensor>>(cnt);
        if (!evalp_list || !opt_list) {
          return Error::MemoryAllocationFailed;
        }
        EValue* none_ev = nullptr;
        for (size_t j = 0; j < cnt; ++j) {
          int32_t vidx = items[j];
          if (vidx == -1) {
            if (!none_ev) {
              void* mem = runtime_alloc->allocateInstance<EValue>();
              if (!mem)
                return Error::MemoryAllocationFailed;
              none_ev = new (mem) EValue();
            }
            evalp_list[j] = none_ev;
          } else if (vidx < 0 || static_cast<size_t>(vidx) >= n) {
            return Error::InvalidProgram;
          } else {
            evalp_list[j] = &d->values[static_cast<size_t>(vidx)];
          }
          new (&opt_list[j]) std::optional<::executorch::aten::Tensor>();
        }
        auto* boxed_mem = runtime_alloc->allocateInstance<
            BoxedEvalueList<std::optional<::executorch::aten::Tensor>>>();
        if (!boxed_mem)
          return Error::MemoryAllocationFailed;
        auto* boxed = new (boxed_mem)
            BoxedEvalueList<std::optional<::executorch::aten::Tensor>>(
                evalp_list, opt_list, cnt);
        d->values[i] = EValue(boxed);
      } break;
      case ValueType::Other:
        d->values[i] = EValue();
        break;
    }
  }
  return Error::Ok;
}

} // namespace native
} // namespace backends
} // namespace executorch
