#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

/*
using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
*/
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
//using executorch::extension::MallocMemoryAllocator;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

int main() {
Result<FileDataLoader> loader =
        FileDataLoader::from("/home/icx-6338/ynimmaga/delegate.pte");
assert(loader.ok());

Result<Program> program = Program::load(&loader.get());
assert(program.ok());

// Method names map back to Python nn.Module method names. Most users will only
// have the singular method "forward".
const char* method_name = "forward";

// MethodMeta is a lightweight structure that lets us gather metadata
// information about a specific method. In this case we are looking to get the
// required size of the memory planned buffers for the method "forward".
Result<MethodMeta> method_meta = program->method_meta(method_name);
assert(method_meta.ok());

std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the Memory
std::vector<Span<uint8_t>> planned_arenas; // Passed to the allocator

size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

// It is possible to have multiple layers in our memory hierarchy; for example,
// SRAM and DRAM.
for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
  // .get() will always succeed because id < num_memory_planned_buffers.
  size_t buffer_size =
      static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
  planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
  planned_arenas.push_back({planned_buffers.back().get(), buffer_size});
}

HierarchicalAllocator planned_memory(
    {planned_arenas.data(), planned_arenas.size()});

// Version of MemoryAllocator that uses malloc to handle allocations rather then
// a fixed buffer.
//MallocMemoryAllocator method_allocator;
MemoryAllocator method_allocator{
     MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

// Assemble all of the allocators into the MemoryManager that the Executor will
// use.
MemoryManager memory_manager(&method_allocator, &planned_memory);

Result<Method> method = program->load_method(method_name);
assert(method.ok());

// Create our input tensor.
float data[1 * 3 * 256 * 256];
Tensor::SizesType sizes[] = {1, 3, 256, 256};
Tensor::DimOrderType dim_order = {0, 1, 2, 3};
TensorImpl impl(
    ScalarType::Float, // dtype
    4, // number of dimensions
    sizes,
    data,
    dim_order);
Tensor t(&impl);

// Implicitly casts t to EValue
Error set_input_error = method->set_input(t, 0);
assert(set_input_error == Error::Ok);

Error execute_error = method->execute();
assert(execute_error == Error::Ok);

EValue output = method->get_output(0);
assert(output.isTensor());

return 0;

}
