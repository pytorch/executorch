This header file `make_boxed_from_unboxed_functor.h` defines a template that can be used to create a boxed version of an unboxed functor. It is part of the executorch extension in the torch namespace.
## Requirements
This header requires C++17 or later.
## Usage
The template takes an unboxed function pointer and wraps it into a functor that takes `KernelRuntimeContext` and `EValues` as inputs and returns void. The wrapped functor will unbox all inputs and forward them to the unboxed kernel.
Here is an example of how to use the template:
```C++
Tensor& my_op(KernelRuntimeContext& ctx, const Tensor& self, const Tensor& other, Tensor& out) {
  // ...
  return out;
}
Kernel my_kernel = Kernel::make_boxed_kernel("my_ns::my_op", EXECUTORCH_FN(my_op));
static auto res = register_kernels({my_kernel});
```
Alternatively, you can use the EXECUTORCH_LIBRARY macro to simplify the process:
```C++
EXECUTORCH_LIBRARY(my_ns, "my_op", my_op);
```
## Details
The template uses a lot of C++17 features to convert each EValue to the inferred argument type. It checks if the first argument is `KernelRuntimeContext`, and if so, it removes it. The call method of the `WrapUnboxedIntoFunctor` struct calls the unboxed function with the corresponding arguments.
The `EXECUTORCH_LIBRARY` macro registers the kernel for the operation and stores the result in a static variable.
## Note
The `KernelRuntimeContext` is a context object that lets kernels handle errors and allocate temp memory. It can be used to add support for other actions in the future.
