# AOTI 符号链接问题修复

## 问题描述

在运行 `gemma3_e2e_runner` 时遇到符号查找错误：

```
./cmake-out/examples/models/gemma3/gemma3_e2e_runner: symbol lookup error: /tmp/token_embedding_so_blob68945.so: undefined symbol: aoti_torch_dtype_bfloat16
```

## 根本原因

1. **AOTI shim 函数定义在头文件中**
   - `/home/gasoonjia/executorch/backends/cuda/runtime/shims/aoti_torch/c/shim.h` 包含了所有 AOTI 函数的定义（包括 `aoti_torch_dtype_bfloat16`）

2. **符号导出宏配置问题**
   - 在 `/home/gasoonjia/executorch/backends/cuda/runtime/shims/aoti_torch/c/macros.h` 中：
   ```c
   #ifdef EXPORT_AOTI_FUNCTIONS
   #define AOTI_TORCH_EXPORT __attribute__((visibility("default")))
   #else
   #define AOTI_TORCH_EXPORT inline
   #endif
   ```

   - 当 `EXPORT_AOTI_FUNCTIONS` **未定义**时，所有函数被标记为 `inline`
   - `inline` 函数不会作为外部符号导出到动态符号表
   - AOT 编译生成的 `.so` 文件（如 `/tmp/token_embedding_so_blob68945.so`）在运行时无法找到这些符号

3. **主程序未导出符号**
   - `gemma3_e2e_runner` 可执行文件需要使用 `--export-dynamic` 链接选项
   - 这样动态加载的库才能访问主程序中的符号

## 解决方案

### 修改 1: 添加 EXPORT_AOTI_FUNCTIONS 编译定义

**文件**: `/home/gasoonjia/executorch/backends/cuda/CMakeLists.txt`

在 `aoti_cuda` 目标添加编译定义：

```cmake
# Define EXPORT_AOTI_FUNCTIONS to export AOTI shim symbols
target_compile_definitions(aoti_cuda PUBLIC EXPORT_AOTI_FUNCTIONS)
```

**作用**:
- 使 `AOTI_TORCH_EXPORT` 宏展开为 `__attribute__((visibility("default")))`
- 所有 AOTI shim 函数将被导出到动态符号表
- AOT 编译的库可以在运行时找到这些符号

### 修改 2: 添加 --export-dynamic 链接选项

**文件**: `/home/gasoonjia/executorch/examples/models/gemma3/CMakeLists.txt`

在 `gemma3_e2e_runner` 目标添加链接选项：

```cmake
# Export dynamic symbols for AOTI runtime linking
if(NOT APPLE)
  target_link_options(gemma3_e2e_runner PRIVATE "LINKER:--export-dynamic")
else()
  target_link_options(gemma3_e2e_runner PRIVATE "LINKER:-export_dynamic")
endif()
```

**作用**:
- 确保主程序的符号在动态符号表中可见
- 动态加载的 `.so` 文件可以解析到这些符号
- 支持跨平台（Linux 和 macOS）

## 如何重新编译

修改后需要重新编译项目：

```bash
# 清理之前的构建（可选但推荐）
rm -rf cmake-out/examples/models/gemma3

# 重新配置 CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_CUDA=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
      -B cmake-out/examples/models/gemma3

# 编译
cmake --build cmake-out/examples/models/gemma3 --target gemma3_e2e_runner --config Release -j
```

**注意**: 如果 VSCode 的 clangd 显示 `shim.h` 中的 "Unknown type name 'AOTI_TORCH_EXPORT'" 等 Hint 级别的诊断信息，这是正常的。这些是因为 clangd 的索引还没有更新。重新编译后这些提示会消失。实际编译时不会有任何问题。

## 验证修复

重新运行 `verify.sh` 或直接运行：

```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path=<your_model_path> \
  --tokenizer_path=<your_tokenizer_path>
```

如果修复成功，应该不再出现 `undefined symbol: aoti_torch_dtype_bfloat16` 错误。

## 技术细节

### 为什么需要 EXPORT_AOTI_FUNCTIONS？

AOT Inductor 生成的代码在运行时需要调用 AOTI runtime 提供的函数。这些函数包括：

- 数据类型相关：`aoti_torch_dtype_*`
- 设备类型相关：`aoti_torch_device_type_*`
- Tensor 操作：`aoti_torch_create_tensor_from_blob`, `aoti_torch_empty_strided`, 等
- 内存管理：`aoti_torch_delete_tensor_object`

这些函数必须在动态符号表中可见，才能被 dlopen 加载的 `.so` 文件解析。

### 为什么需要 --export-dynamic？

Linux 下默认情况，可执行文件的符号不会被导出到动态符号表。`--export-dynamic` 选项：

1. 将所有全局符号添加到动态符号表
2. 允许 dlopen 加载的库通过 dlsym 查找这些符号
3. 在 runtime 建立正确的符号链接

## 相关文件

- `/home/gasoonjia/executorch/backends/cuda/runtime/shims/aoti_torch/c/shim.h` - AOTI 函数定义
- `/home/gasoonjia/executorch/backends/cuda/runtime/shims/aoti_torch/c/macros.h` - 导出宏定义
- `/home/gasoonjia/executorch/backends/cuda/runtime/cuda_backend.cpp` - CUDA backend 实现
- `/home/gasoonjia/executorch/backends/cuda/CMakeLists.txt` - CUDA backend 构建配置
- `/home/gasoonjia/executorch/examples/models/gemma3/CMakeLists.txt` - Gemma3 runner 构建配置
