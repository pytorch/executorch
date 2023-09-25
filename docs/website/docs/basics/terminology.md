# Basics
## Terminology

### ATen mode
ATen mode uses the ATen (pytorch core) implementation of Tensor (`at::Tensor`)
along with related types (ScalarType, etc.)
* `at::Tensor` is big and complex, and often allocates memory with new/malloc
* The ATen kernels, which rely on the full `at::Tensor` API, are usable in this
  configuration
* Those kernels also tend to do dynamic memory allocation, and often have extra
  flexibility (and thus overhead) to handle things not needed by mobile/embedded
  clients: e.g., CUDA support, sparse tensor support, dtype promotion

### Portable mode
Portable mode uses ExecuTorch's smaller `torch::executor::Tensor` (aka ETensor)
implementation, along with related types (`torch::executor::ScalarType`, etc.)
* ETensor's API is a source-compatible subset of `at::Tensor`. Code that is
  written against ETensor can also build against `at::Tensor`.
* "portable mode kernels" are any operator implementations that are written to be
  compatible with ETensor. But that means that the can also build against
  `at::Tensor` if desired, and used in the same model as ATen kernels.
* ETensor does not own or allocate memory on its own
  * To support dynamic shapes, kernels can allocate
    Tensor data using the MemoryAllocator provided by the client.

### Portable kernels
See (`//executorch/kernels/portable/README.md`) for technical details.
Portable kernels, which live under `//executorch/kernels/portable`, are:
* Portable mode kernels
* Compatible with ATen operator signatures
* Written in portable C++ so that they can build for any target
* Written as reference implementations, prioritizing clarity and simplicity
  over optimization
* Generally much smaller in code size than ATen kernels
* Written to avoid dynamically allocating memory using new/malloc
  * To support dynamic shapes, some kernels may allocate
    Tensor data using the MemoryAllocator provided by the client.
