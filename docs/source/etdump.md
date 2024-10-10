# Prerequisite | ETDump - ExecuTorch Dump

ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch Developer Tools. It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime. Users can't parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data, offering interfaces for flexible analysis and debugging.


## Generating an ETDump

Generating an ETDump is a relatively straightforward process. Users can follow the steps detailed below to integrate it into their application that uses ExecuTorch.

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

4. ***Compile*** your binary using CMake with the `ET_EVENT_TRACER_ENABLED` pre-processor flag to enable events to be traced and logged into ETDump inside the ExecuTorch runtime. This flag needs to be added to the ExecuTorch library and any operator library that you are compiling into your binary. For reference, you can take a look at `examples/sdk/CMakeLists.txt`. The lines of interest are:
```
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
target_compile_options(portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED)
```
## Using an ETDump

Pass this ETDump into the [Inspector API](./model-inspector.rst) to access this data and do post-run analysis.
