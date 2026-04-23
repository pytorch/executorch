# Prerequisite | ETDump - ExecuTorch Dump

ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch Developer Tools. It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime. Users can't parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data, offering interfaces for flexible analysis and debugging.


## Generating an ETDump

Generating an ETDump is a relatively straightforward process. Users can follow the steps detailed below to integrate it into their application that uses ExecuTorch.

### Build Configuration

To enable ETDump support, you need to configure your CMake build with the following options:

**Required CMake options:**
```cmake
-DEXECUTORCH_BUILD_DEVTOOLS=ON     # Builds the devtools library including ETDump
-DEXECUTORCH_ENABLE_EVENT_TRACER=ON # Enables event tracing in the runtime
```

**Required CMake targets to link:**
```cmake
target_link_libraries(your_binary
    etdump       # ETDump library
    # ... other ExecuTorch libraries
)
```

The `EXECUTORCH_ENABLE_EVENT_TRACER=ON` option automatically sets the `ET_EVENT_TRACER_ENABLED` preprocessor flag for the ExecuTorch runtime and all operator libraries.

### Using the Low-Level Runtime API

1. ***Include*** the ETDump header in your code.
```C++
#include <executorch/devtools/etdump/etdump_flatcc.h>
```

2. ***Create*** an Instance of the ETDumpGen class and pass it into the `load_method` call that is invoked in the runtime.

```C++
executorch::etdump::ETDumpGen etdump_gen;
Result<Method> method =
      program->load_method(method_name, &memory_manager, &etdump_gen);
```

3. ***Dump Out the ETDump Buffer*** - after the inference iterations have been completed, users can dump out the ETDump buffer. If users are on a device which has a filesystem, they could just write it out to the filesystem. For more constrained embedded devices, users will have to extract the ETDump buffer from the device through a mechanism that best suits them (e.g. UART, JTAG etc.)

```C++
etdump_result result = etdump_gen.get_etdump_data();
if (result.buf != nullptr && result.size > 0) {
    // On a device with a file system users can just write it out
    // to the file-system.
    FILE* f = fopen(FLAGS_etdump_path.c_str(), "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  }
```

### Using the Module API (C++)

For applications using the higher-level [Module API](extension-module.md), ETDump integration is simpler. Create an `ETDumpGen` instance and pass it to the `Module` constructor:

```cpp
#include <fstream>
#include <memory>

#include <executorch/extension/module/module.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>

using namespace ::executorch::extension;

Module module("/path/to/model.pte", Module::LoadMode::Mmap, std::make_unique<ETDumpGen>());

// Execute a method, e.g., module.forward(...); or module.execute("my_method", ...);

if (auto* etdump = dynamic_cast<ETDumpGen*>(module.event_tracer())) {
  const auto trace = etdump->get_etdump_data();

  if (trace.buf && trace.size > 0) {
    std::unique_ptr<void, decltype(&free)> guard(trace.buf, free);
    std::ofstream file("/path/to/trace.etdump", std::ios::binary);

    if (file) {
      file.write(static_cast<const char*>(trace.buf), trace.size);
    }
  }
}
```

### Using the Python Runtime API

For Python applications using the [Python Runtime API](runtime-python-api-reference.rst), you can enable ETDump when loading a program:

```python
from pathlib import Path
import os

import torch
from executorch.runtime import Runtime, Program, Method

# Create program with etdump generation enabled
et_runtime: Runtime = Runtime.get()
program: Program = et_runtime.load_program(
    Path("/tmp/program.pte"),
    enable_etdump=True,
    debug_buffer_size=int(1e7),  # 10MB buffer to capture all debug info
)

# Load method and execute
forward: Method = program.load_method("forward")
inputs = (torch.ones(2, 2), torch.ones(2, 2))
outputs = forward.execute(inputs)

# Write etdump result to file
etdump_file = "/tmp/etdump_output.etdp"
debug_file = "/tmp/debug_output.bin"
program.write_etdump_result_to_file(etdump_file, debug_file)

# Check that files were created
print(f"ETDump file created: {os.path.exists(etdump_file)}")
print(f"Debug file created: {os.path.exists(debug_file)}")
```

**Note:** The Python Runtime API requires ExecuTorch to be built with event tracing enabled (`EXECUTORCH_ENABLE_EVENT_TRACER=ON`).

### Troubleshooting: Empty ETDump

If the binary is not compiled with the `ET_EVENT_TRACER_ENABLED` preprocessor flag (either by setting `EXECUTORCH_ENABLE_EVENT_TRACER=ON` in CMake or manually adding `-DET_EVENT_TRACER_ENABLED`), no trace events will be recorded and the ETDump will be empty.

When this flag is missing, the following code:

```C++
ETDumpResult result = etdump_gen.get_etdump_data();
if (result.buf != nullptr && result.size > 0) {
    ...
}
```

will always return:

```C++
result.buf == nullptr
result.size == 0
```

This indicates that no ETDump data was collected, and therefore nothing can be analyzed through the Inspector API.

## Using an ETDump

Pass this ETDump into the [Inspector API](model-inspector.rst) to access this data and do post-run analysis.
