# Prerequisite | ETDump - ExecuTorch Dump

ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch SDK experience. It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime. Users can't parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data, offering interfaces for flexible analysis and debugging.


## Generating an ETDump

Generating an ETDump is a relatively straightforward process. Users can follow the steps detailed below to integrate it into their application that uses ExecuTorch.

1. ***Include*** the ETDump header in your code.
```C++
#include <executorch/sdk/etdump/etdump_flatcc.h>
```

2. ***Create*** an Instance of the ETDumpGen class and pass it into the `load_method` call that is invoked in the runtime.

```C++
torch::executor::ETDumpGen etdump_gen = torch::executor::ETDumpGen();
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

4. ***Compile*** your binary with the `ET_EVENT_TRACER_ENABLED` pre-processor flag to enable events to be traced and logged into ETDump inside the ExecuTorch runtime.

    i). ***Buck***

    In Buck, users simply depend on the etdump target which is:
    ```
    //executorch/sdk/etdump:etdump_flatcc
    ```
    When compiling their binary through Buck, users can pass in this buck config to enable the pre-processor flag. For example, when compiling `sdk_example_runner` to enable ETDump generation, users compile using the following command:
    ```
    buck2 build -c executorch.event_tracer_enabled=true examples/sdk/sdk_example_runner:sdk_example_runner
    ```

    ii). ***CMake***

    In CMake, users add this to their compile flags:
    ```
    -DET_EVENT_TRACER_ENABLED
    ```

    This flag needs to be added to the ExecuTorch library and any operator library that the users are compiling into their binary. For reference, users can take a look at `examples/sdk/CMakeLists.txt`. The lines of of interest are:
    ```
    target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)
    target_compile_options(portable_ops_lib PUBLIC -DET_EVENT_TRACER_ENABLED)
    ```
## Using an ETDump

1. Pass this ETDump into the [Inspector API](./sdk-inspector.rst) to access this data and  do post-run analysis.
