# ETDump - ExecuTorch Dump

ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch SDK experience. It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime. Users can't parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data, offering interfaces for flexible analysis and debugging.


## Generating an ETDump:

Generating an ETDump is a relatively straight forward process. Users can follow the steps detailed below to integrate it into their application that uses ExecuTorch.

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

3. ***Dump out the ETDump buffer*** - after the inference iterations have been completed, users can dump out the ETDump buffer. If users are on a device which has a file-system, they could just write it out to the fileystem. For more constrained embedded devices, users will have to extract the ETDump buffer from the device through a mechanism that best suits them (e.g. UART, JTAG etc.)

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

4. ***Compile*** your binary with the flags that enable events to be traced and logged into ETDump inside the ExecuTorch runtime. The pre-processor flag that controls this is `ET_EVENT_TRACER_ENABLED`.

    i). ***CMake***

    In CMake users can add this to their compile flags:
    ```
    -DET_EVENT_TRACER_ENABLED
    ```

    ii). ***Buck***

    In Buck users can simply depend on the etdump target which is:
    ```
    //executorch/sdk/etdump:etdump_flatcc
    ```
    When compiling their binary through Buck, users can pass in this buck config which will enable this pre-processor flag:
    ```
    buck build -c executorch.event_tracer_enabled=true your_binary_target
    ```

    TODO : Point to sample runner in examples here.

## Using an ETDump:

1. Pass this ETDump into the [Inspector API](./sdk-inspector.rst) for access to this data and to do post-run analysis on this data.
