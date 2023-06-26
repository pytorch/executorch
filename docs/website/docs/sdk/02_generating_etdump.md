# Generating an ETdump
> Note: If you're using AIBench with one of the pre-built binaries then you can skip this step.
AIBench will handle the retrieval and processing of ETdump for you.

Generating an ETdump from the runtime is very trivial. It consists of two steps.

1) Integrate the etdump generation code into your application to generate an etdump after the model has been executed.

2) Compile your binary with the profiling enabled flag to ensure that the profiling data is written out to etdump.

---

####  1. Integrating ETdump generation code into your application

- Add this header to your file
```c
#include <executorch/sdk/etdump/etdump.h>
```

- Create an instance of `ETDump` by passing in a `MemoryAllocator`.

```c
MemoryAllocator etdump_allocator{MemoryAllocator(ETDumpMemPoolSize, etdump_mem_pool)};}
ETDump et_dump(etdump_allocator);
```

- Write out the `ETDump` to the filesystem.

```c
auto ret = et_dump.serialize_prof_results_to_etdump(etdump_path)
if (ret != torch::executor::Error::Ok) {
    ET_LOG(Error, "Failed to serialize and write out etdump data.");
    return -1;
    }
}
```

#### 2. Generating profiling data into an ETdump

To make sure that the profiling data collected in the runtime is written out to the ETDump your application will have to be compiled with the following flag.

```c
-DPROFILING_ENABLED
```

In buck this is as simple as compiling your target with the following buck config:
```
-c executorch.prof_enabled=true
```
