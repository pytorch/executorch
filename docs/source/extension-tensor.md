# Managing Tensor Memory in C++

**Author:** [Anthony Shoumikhin](https://github.com/shoumikhin)

Tensors are fundamental data structures in ExecuTorch, representing multi-dimensional arrays used in computations for neural networks and other numerical algorithms. In ExecuTorch, the [Tensor](https://github.com/pytorch/executorch/blob/main/runtime/core/portable_type/tensor.h) class doesn’t own its metadata (sizes, strides, dim_order) or data, keeping the runtime lightweight. Users are responsible for supplying all these memory buffers and ensuring that the metadata and data outlive the `Tensor` instance. While this design is lightweight and flexible, especially for tiny embedded systems, it places a significant burden on the user. However, if your environment requires minimal dynamic allocations, a small binary footprint, or limited C++ standard library support, you’ll need to accept that trade-off and stick with the regular `Tensor` type.

Imagine you’re working with a [`Module`](extension-module.md) interface, and you need to pass a `Tensor` to the `forward()` method. You would need to declare and maintain at least the sizes array and data separately, sometimes the strides too, often leading to the following pattern:

```cpp
#include <executorch/extension/module/module.h>

using namespace executorch::aten;
using namespace executorch::extension;

SizesType sizes[] = {2, 3};
DimOrderType dim_order[] = {0, 1};
StridesType strides[] = {3, 1};
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
TensorImpl tensor_impl(
    ScalarType::Float,
    std::size(sizes),
    sizes,
    data,
    dim_order,
    strides);
// ...
module.forward(Tensor(&tensor_impl));
```

You must ensure `sizes`, `dim_order`, `strides`, and `data` stay valid. This makes code maintenance difficult and error-prone. Users have struggled to manage lifetimes, and many have created their own ad-hoc managed tensor abstractions to hold all the pieces together, leading to a fragmented and inconsistent ecosystem.

## Introducing TensorPtr

To alleviate these issues, ExecuTorch provides `TensorPtr` and `TensorImplPtr` via the new [Tensor Extension](https://github.com/pytorch/executorch/tree/main/extension/tensor) that manage the lifecycle of tensors and their implementations. These are essentially smart pointers (`std::unique_ptr<Tensor>` and `std::shared_ptr<TensorImpl>`, respectively) that handle the memory management of both the tensor's data and its dynamic metadata.

Now, users no longer need to worry about metadata lifetimes separately. Data ownership is determined based on whether the it is passed by pointer or moved into the `TensorPtr` as an `std::vector`. Everything is bundled in one place and managed automatically, enabling you to focus on actual computations.

Here’s how you can use it:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::extension;

auto tensor = make_tensor_ptr(
    {2, 3},                                // sizes
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // data
// ...
module.forward(tensor);
```

The data is now owned by the tensor instance because it's provided as a vector. To create a non-owning `TensorPtr` just pass the data by pointer. The `type` is deduced automatically from the data vector (`float`). `strides` and `dim_order` are computed automatically to the default values based on the `sizes` if not specified explicitly as extra arguments.

`EValue` in `Module::forward()` accepts `TensorPtr` directly, ensuring seamless integration. `EValue` can now be constructed implicitly with a smart pointer to any type that it can hold, so `TensorPtr` gets dereferenced implicitly and `EValue` holding a `Tensor` that the  `TensorPtr` pointed at is passed to the `forward()`.

## API Overview

The new API revolves around two main smart pointers:

- `TensorPtr`: `std::unique_ptr` managing a `Tensor` object. Since each `Tensor` instance is unique, `TensorPtr` ensures exclusive ownership.
- `TensorImplPtr`: `std::shared_ptr` managing a `TensorImpl` object. Multiple `Tensor` instances can share the same `TensorImpl`, so `TensorImplPtr` uses shared ownership.

### Creating Tensors

There are several ways to create a `TensorPtr`.

### Creating Scalar Tensors

You can create a scalar tensor, i.e. a tensor with zero dimensions or with one of sizes being zero.

*Providing A Single Data Value*

```cpp
auto tensor = make_tensor_ptr(3.14);
```

The resulting tensor will contain a single value 3.14 of type double, which is deduced automatically.

*Providing A Single Data Value with a Type*

```cpp
auto tensor = make_tensor_ptr(42, ScalarType::Float);
```

Now the integer 42 will be cast to float and the tensor will contain a single value 42 of type float.

#### Owning a Data Vector

When you provide sizes and data vectors, `TensorPtr` takes ownership of both the data and the sizes.

*Providing Data Vector*

```cpp
auto tensor = make_tensor_ptr(
    {2, 3},                                 // sizes
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // data (float)
```

The type is deduced automatically as `ScalarType::Float` from the data vector.

*Providing Data Vector with a Type*

If you provide data of one type but specify a different scalar type, the data will be cast to the specified type.

```cpp
auto tensor = make_tensor_ptr(
    {1, 2, 3, 4, 5, 6},          // data (int)
    ScalarType::Double);         // double scalar type
```

In this example, even though the data vector contains integers, we specify the scalar type as `Double`. The integers are cast to doubles, and the new data vector is owned by the `TensorPtr`. The `sizes` argument is skipped in this example, so the input data vector's size is used. Note that we forbid the opposite cast, when a floating point type casts to an integral type, because that loses precision. Similarly, casting other types to `Bool` isn't allowed.

*Providing Data Vector as `std::vector<uint8_t>`*

You can also provide raw data as a `std::vector<uint8_t>`, specifying the sizes and scalar type. The data will be reinterpreted according to the provided type.

```cpp
std::vector<uint8_t> data = /* raw data */;
auto tensor = make_tensor_ptr(
    {2, 3},                 // sizes
    std::move(data),        // data as uint8_t vector
    ScalarType::Int);       // int scalar type
```

The `data` vector must be large enough to accommodate all the elements according to the provided sizes and scalar type.

#### Non-Owning a Raw Data Pointer

You can create a `TensorPtr` that references existing data without taking ownership.

*Providing Raw Data*

```cpp
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto tensor = make_tensor_ptr(
    {2, 3},              // sizes
    data,                // raw data pointer
    ScalarType::Float);  // float scalar type
```

The `TensorPtr` does not own the data, you must ensure the `data` remains valid.

*Providing Raw Data with Custom Deleter*

If you want `TensorPtr` to manage the lifetime of the data, you can provide a custom deleter.

```cpp
auto* data = new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
auto tensor = make_tensor_ptr(
    {2, 3},                               // sizes
    data,                                 // data pointer
    ScalarType::Double,                   // double scalar type
    TensorShapeDynamism::DYNAMIC_BOUND,   // some default dynamism
    [](void* ptr) { delete[] static_cast<double*>(ptr); });
```

The `TensorPtr` will call the custom deleter when it is destroyed, i.e. when the smart pointer is reset and no more references to the underlying `TensorImplPtr` exist.

#### Sharing Existing Tensor

You can create a `TensorPtr` by wrapping an existing `TensorImplPtr`, and the latter can be created with the same collection of APIs as `TensorPtr`. Any changes made to `TensorImplPtr` or any `TensorPtr` sharing the same `TensorImplPtr` get reflected in for all.

*Sharing Existing TensorImplPtr*

```cpp
auto tensor_impl = make_tensor_impl_ptr(
    {2, 3},
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
auto tensor = make_tensor_ptr(tensor_impl);
auto tensor_copy = make_tensor_ptr(tensor_impl);
```

Both `tensor` and `tensor_copy` share the underlying `TensorImplPtr`, reflecting changes in data but not in metadata.

Also, you can create a new `TensorPtr` that shares the same `TensorImplPtr` as an existing `TensorPtr`.

*Sharing Existing TensorPtr*

```cpp
auto tensor_copy = make_tensor_ptr(tensor);
```

#### Viewing Existing Tensor

You can create a `TensorPtr` from an existing `Tensor`, copying its properties and referencing the same data.

*Viewing Existing Tensor*

```cpp
Tensor original_tensor = /* some existing tensor */;
auto tensor = make_tensor_ptr(original_tensor);
```

Now the newly created `TensorPtr` references the same data as the original tensor, but has its own metadata copy, so can interpret or "view" the data differently, but any modifications to the data will be reflected for the original `Tensor` too.

### Cloning Tensors

To create a new `TensorPtr` that owns a copy of the data from an existing tensor:

```cpp
Tensor original_tensor = /* some existing tensor */;
auto tensor = clone_tensor_ptr(original_tensor);
```

The newly created `TensorPtr` has its own copy of the data, so can modify and manage it independently.
Likewise, you can create a clone of an existing `TensorPtr`.

```cpp
auto original_tensor = make_tensor_ptr();
auto tensor = clone_tensor_ptr(original_tensor);
```

Note that regardless of whether the original `TensorPtr` owns the data or not, the newly created `TensorPtr` will own a copy of the data.

### Resizing Tensors

The `TensorShapeDynamism` enum specifies the mutability of a tensor's shape:

- `STATIC`: The tensor's shape cannot be changed.
- `DYNAMIC_BOUND`: The tensor's shape can be changed, but can never contain more elements than it had at creation based on the initial sizes.
- `DYNAMIC`: The tensor's shape can be changed arbitrarily. Note that currently `DYNAMIC` is an alias of `DYNAMIC_BOUND`.

When resizing a tensor, you must respect its dynamism setting. Resizing is only allowed for tensors with `DYNAMIC` or `DYNAMIC_BOUND` shapes, and you cannot resize `DYNAMIC_BOUND` tensor to contain more elements than it had initially.

```cpp
auto tensor = make_tensor_ptr(
    {2, 3},                      // sizes
    {1, 2, 3, 4, 5, 6},          // data
    ScalarType::Int,
    TensorShapeDynamism::DYNAMIC_BOUND);
// Initial sizes: {2, 3}
// Number of elements: 6

resize_tensor_ptr(tensor, {2, 2});
// The tensor's sizes are now {2, 2}
// Number of elements is 4 < initial 6

resize_tensor_ptr(tensor, {1, 3});
// The tensor's sizes are now {1, 3}
// Number of elements is 3 < initial 6

resize_tensor_ptr(tensor, {3, 2});
// The tensor's sizes are now {3, 2}
// Number of elements is 6 == initial 6

resize_tensor_ptr(tensor, {6, 1});
// The tensor's sizes are now {6, 1}
// Number of elements is 6 == initial 6
```

## Convenience Helpers

ExecuTorch provides several helper functions to create tensors conveniently.

### Creating Non-Owning Tensors with `for_blob` and `from_blob`

These helpers allow you to create tensors that do not own the data.

*Using `from_blob()`*

```cpp
float data[] = {1.0f, 2.0f, 3.0f};
auto tensor = from_blob(
    data,                // data pointer
    {3},                 // sizes
    ScalarType::Float);  // float scalar type
```

*Using `for_blob()` with Fluent Syntax*

```cpp
double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
auto tensor = for_blob(data, {2, 3}, ScalarType::Double)
                  .strides({3, 1})
                  .dynamism(TensorShapeDynamism::STATIC)
                  .make_tensor_ptr();
```

*Using Custom Deleter with `from_blob()`*

```cpp
int* data = new int[3]{1, 2, 3};
auto tensor = from_blob(
    data,             // data pointer
    {3},              // sizes
    ScalarType::Int,  // int scalar type
    [](void* ptr) { delete[] static_cast<int*>(ptr); });
```

The `TensorPtr` will call the custom deleter when it is destroyed.

### Creating Empty Tensors

`empty()` creates an uninitialized tensor with sizes specified.

```cpp
auto tensor = empty({2, 3});
```

`empty_like()` creates an uninitialized tensor with the same sizes as an existing `TensorPtr`.

```cpp
TensorPtr original_tensor = /* some existing tensor */;
auto tensor = empty_like(original_tensor);
```

And `empty_strided()` creates an uninitialized tensor with sizes and strides specified.

```cpp
auto tensor = empty_strided({2, 3}, {3, 1});
```

### Creating Tensors Filled with Specific Values

`full()`, `zeros()` and `ones()` create a tensor filled with a provided value, zeros or ones respectively.

```cpp
auto tensor_full = full({2, 3}, 42.0f);
auto tensor_zeros = zeros({2, 3});
auto tensor_ones = ones({3, 4});
```

Similarly to `empty()`, there are extra helper functions `full_like()`, `full_strided()`, `zeros_like()`, `zeros_strided()`, `ones_like()` and `ones_strided()` to create filled tensors with the same properties as an existing `TensorPtr` or with custom strides.

### Creating Random Tensors

`rand()` creates a tensor filled with random values between 0 and 1.

```cpp
auto tensor_rand = rand({2, 3});
```

`randn()` creates a tensor filled with random values from a normal distribution.

```cpp
auto tensor_randn = randn({2, 3});
```

`randint()` creates a tensor filled with random integers between min (inclusive) and max (exclusive) integers specified.

```cpp
auto tensor_randint = randint(0, 10, {2, 3});
```

### Creating Scalar Tensors

In addition to `make_tensor_ptr()` with a single data value, you can create a scalar tensor with `scalar_tensor()`.

```cpp
auto tensor = scalar_tensor(3.14f);
```

Note that the `scalar_tensor()` function expects a value of type `Scalar`. In ExecuTorch, `Scalar` can represent `bool`, `int`, or floating-point types, but not types like `Half` or `BFloat16`, etc. for which you'd need to use `make_tensor_ptr()` to skip the `Scalar` type.

## Notes on EValue and Lifetime Management

The [`Module`](extension-module.md) interface expects data in the form of `EValue`, a variant type that can hold a `Tensor` or other scalar types. When you pass a `TensorPtr` to a function expecting an `EValue`, you can dereference the `TensorPtr` to get the underlying `Tensor`.

```cpp
TensorPtr tensor = /* create a TensorPtr */
//...
module.forward(tensor);
```

Or even a vector of `EValues` for multiple parameters.

```cpp
TensorPtr tensor = /* create a TensorPtr */
TensorPtr tensor2 = /* create another TensorPtr */
//...
module.forward({tensor, tensor2});
```

However, be cautious: `EValue` will not hold onto the dynamic data and metadata from the `TensorPtr`. It merely holds a regular `Tensor`, which does not own the data or metadata but refers to them using raw pointers. You need to ensure that the `TensorPtr` remains valid for as long as the `EValue` is in use.

This also applies when using functions like `set_input()` or `set_output()` that expect `EValue`.

## Interoperability with ATen

If your code is compiled with the preprocessor flag `USE_ATEN_LIB` turned on, all the `TensorPtr` APIs will use `at::` APIs under the hood. E.g. `TensorPtr` becomes a `std::unique_ptr<at::Tensor>` and `TensorImplPtr` becomes `c10::intrusive_ptr<at::TensorImpl>`. This allows for seamless integration with [PyTorch ATen](https://pytorch.org/cppdocs) library.

### API Equivalence Table

Here's a table matching `TensorPtr` creation functions with their corresponding ATen APIs:

| ATen                                        | ExecuTorch                                  |
|---------------------------------------------|---------------------------------------------|
| `at::tensor(data, type)`                    | `make_tensor_ptr(data, type)`               |
| `at::tensor(data, type).reshape(sizes)`     | `make_tensor_ptr(sizes, data, type)`        |
| `tensor.clone()`                            | `clone_tensor_ptr(tensor)`                  |
| `tensor.resize_(new_sizes)`                 | `resize_tensor_ptr(tensor, new_sizes)`      |
| `at::scalar_tensor(value)`                  | `scalar_tensor(value)`                      |
| `at::from_blob(data, sizes, type)`          | `from_blob(data, sizes, type)`              |
| `at::empty(sizes)`                          | `empty(sizes)`                              |
| `at::empty_like(tensor)`                    | `empty_like(tensor)`                        |
| `at::empty_strided(sizes, strides)`         | `empty_strided(sizes, strides)`             |
| `at::full(sizes, value)`                    | `full(sizes, value)`                        |
| `at::full_like(tensor, value)`              | `full_like(tensor, value)`                  |
| `at::full_strided(sizes, strides, value)`   | `full_strided(sizes, strides, value)`       |
| `at::zeros(sizes)`                          | `zeros(sizes)`                              |
| `at::zeros_like(tensor)`                    | `zeros_like(tensor)`                        |
| `at::zeros_strided(sizes, strides)`         | `zeros_strided(sizes, strides)`             |
| `at::ones(sizes)`                           | `ones(sizes)`                               |
| `at::ones_like(tensor)`                     | `ones_like(tensor)`                         |
| `at::ones_strided(sizes, strides)`          | `ones_strided(sizes, strides)`              |
| `at::rand(sizes)`                           | `rand(sizes)`                               |
| `at::rand_like(tensor)`                     | `rand_like(tensor)`                         |
| `at::randn(sizes)`                          | `randn(sizes)`                              |
| `at::randn_like(tensor)`                    | `randn_like(tensor)`                        |
| `at::randint(low, high, sizes)`             | `randint(low, high, sizes)`                 |
| `at::randint_like(tensor, low, high)`       | `randint_like(tensor, low, high)`           |

## Best Practices

- *Manage Lifetimes Carefully*: Even though `TensorPtr` and `TensorImplPtr` handle memory management, you still need to ensure that any non-owned data (e.g., when using `from_blob()`) remains valid while the tensor is in use.
- *Use Convenience Functions*: Utilize the provided helper functions for common tensor creation patterns to write cleaner and more readable code.
- *Be Aware of Data Ownership*: Know whether your tensor owns its data or references external data to avoid unintended side effects or memory leaks.
- *Ensure TensorPtr Outlives EValue*: When passing tensors to modules that expect `EValue`, make sure the `TensorPtr` remains valid as long as the `EValue` is in use.
- *Understand Scalar Types*: Be mindful of the scalar types when creating tensors, especially when casting between types.

## Conclusion

The `TensorPtr` and `TensorImplPtr` in ExecuTorch simplifies tensor memory management by bundling the data and dynamic metadata into smart pointers. This design eliminates the need for users to manage multiple pieces of data and ensures safer and more maintainable code.

By providing interfaces similar to PyTorch's ATen library, ExecuTorch makes it easier for developers to adopt the new API without a steep learning curve.
