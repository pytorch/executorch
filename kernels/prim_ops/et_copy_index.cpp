/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/prim_ops/et_copy_index.h>

#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using exec_aten::SizesType;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::resize_tensor;

namespace torch {
namespace executor {
namespace function {

constexpr size_t kTensorDimensionLimit = 16;

// This operator is currently only intended for use to support the map operator.
// Below is a model with the map operator in it.
// def map_fn(x,y):
//    return x+y
//
// class TestMapCond(torch.nn.Module):
//    def __init__(self):
//        super().__init__()
//
//    def forward(self, x,y):
//        return control_flow.map(map_fn, x, y)
//
// Corresponding graph:
//    def forward(self, arg0_1, arg1_1):
//        submodule_0 = self.submodule_0
//        map_1 = torch.ops.map(submodule_0, arg0_1, arg1_1);  submodule_0 =
//        arg0_1 = arg1_1 = None return [map_1]
//
//    def forward(self, arg0_1, arg1_1):
//        add_tensor = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 =
//        arg1_1 = None
//        return add_tensor
// Post the transformations by the emitter to handle the map loop this is what
// the submodule that map calls will look like.
//   def forward(self, arg0_1, arg1_1):
//        sym_size = torch.ops.aten.sym_size(arg0_1)
//        # Emitter creates a variable here to track iteration index
//        select_copy_tensor = torch.ops.aten.select(arg0_1, 0, iteration_index)
//        add_tensor = torch.ops.aten.add.Tensor(select_copy_tensor, arg1_1);
//        arg0_1 = arg1_1 = None output_of_map =
//        torch.ops.executorch.prim.et_copy_index(output_of_map, add_tensor,
//        iteration_index) iteration_index =
//        torch.ops.executorch.prim.add.int(iteration_index, 1, iteration_index)
//        done_bool = torch.ops.executorch.prim.eq.int(iteration_index,
//        sym_size, done_bool) # Emitter inserts a instruction here, if
//        done_bool == False jump to selcect_copy op # if not continue. return
//        add_tensor
//
// The output of each iteration (copy_from) is copied into the copy_to tensor at
// the specified index. This operator is supported in both ATen and lean modes.
void et_copy_index(KernelRuntimeContext& context, EValue** stack) {
  (void)context;
  SizesType expected_output_size[kTensorDimensionLimit];

  auto copy_to = (*stack[0]).toTensor();
  auto copy_from = (*stack[1]).toTensor();
  auto index = (*stack[2]).toInt();

  // Number of bytes we need to copy over from copy_from tensor.
  size_t size_copy_from = (copy_from.element_size()) * (copy_from.numel());

  ET_CHECK_MSG(
      (copy_to.sizes().size() - copy_from.sizes().size()) == 1,
      "Ranks of copy_to  and copy_from tensor should only differ by 1.");

  // Here we calculate the size of the out_tensor after copy_from has
  // been copied to it. This will be passed onto the resize call.
  expected_output_size[0] = index + 1;
  for (size_t i = 0; i < copy_from.sizes().size(); i++) {
    // If we're copying past the first index then the shape of
    // copy_from and copy_to without the leading dimension should be
    // the same. i.e. copy_to.size[1:] == copy_from.size[:].
    if (index > 0) {
      ET_CHECK_MSG(
          copy_to.sizes()[i + 1] == copy_from.sizes()[i],
          "Mismatch in shape between copy_to and copy_from tensors");
    }
    expected_output_size[i + 1] = copy_from.sizes()[i];
  }

  if (copy_to.sizes()[0] < expected_output_size[0]) {
    // Resize `copy_to` to the expected output size.
    const void* data_ptr = copy_to.const_data_ptr();
    Error err =
        resize_tensor(copy_to, {expected_output_size, copy_to.sizes().size()});
    ET_CHECK(err == Error::Ok);
    ET_CHECK_MSG(
        data_ptr == copy_to.const_data_ptr(),
        "Data ptr of copy_to tensor changed after resize which isn't allowed for static/upper-bounded tensors");
  }

  auto copy_to_ptr = copy_to.const_data_ptr();
  auto copy_from_ptr = copy_from.const_data_ptr();

  // If we've reached here, it means the copy_to tensor has been
  // successfully resized so we can now copy over the data from
  // copy_from into the copy_to tensor.
  memcpy(
      (void*)((uintptr_t)copy_to_ptr + index * size_copy_from),
      copy_from_ptr,
      size_copy_from);
}

} // namespace function
} // namespace executor
} // namespace torch
