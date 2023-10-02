# Profiling in ExecuTorch

## Introduction
Profiling in ExecuTorch is broken down into two stages.

1) Running your test binary/application and collecting the profiling buffer dump.
2) Using the post-processing library/tool to generate the profiling metrics from the  profiling buffer dump.

## Important concepts
There are two main concepts to understand in ExecuTorch profiling which are profiling blocks and events. We elaborate on what these are below:

### Profiling Events
A profiling event can be either of these:
- Code execution (time taken for execution) event
- Memory allocation event

For code execution events it's required that the user explictily marks the segment of the code that they want profiled. (There are a set of core events that are already profiled out of the box which will be described later).

For memory allocation events, the user will be explictily required to enable profiling on their memory allocator. The interface for this will be described later in the API section.

### Profiling Blocks
A profiling block is essentially a block of code that consists of a set of profiling events that take place within it. The user can decide to mark different parts of their code as different profiling blocks by tagging them with a certain name. During post-processing all the metrics for profiling blocks with the same name will be aggregated and presented to the user. It's easy to derive the p10, p90, mean, median etc. from these aggregated values. The API to create new profiling blocks will be described below in the API section.

## Build time configuration

There are 3 pre-processor flags that are of importance to profiling in ExecuTorch. These are:
- **PROFILING_ENABLED** - Passing in this flag enables profliling in ExecuTorch and all profiling events will be logged to the proifling buffer. If this flag isn't passed in then all the profiling hooks that have been placed in the code (including the core runtime) will all resolve to no-ops and have no effect on the binary size.
- **MAX_PROFILE_EVENTS** - This flag sets the maximum number of events that can be logged into a profiling block. If not passed in the default value resolves to 1024
- **MAX_PROFILE_BLOCKS** - This flag sets the maximum number of profiling blocks supported. If not passed in the default value resolves to 2.

While building your target application with Buck the above pre-processor flags can be controlled by these Buck configs.

| Pre-processor flag       | Buck config                                            |
| -------------------------| ----------- |
| `PROFILING_ENABLED`      | `-c executorch.prof_enabled=\<true,false>`             |
| `MAX_PROFILE_EVENTS`     | `-c executorch.prof_buf_size=<max events>`             |
| `MAX_PROFILE_BLOCKS`.    | `-c executorch.num_prof_blocks=<max profiling blocks>` |

***Important Note***: When `PROFILING_ENABLED` is not passed in none of the profiling hooks will be enabled as they will all resolve to no-ops and there will be no impact on the binary size of the resulting target.

## Events tracked by the core runtime
In the core runtime we already have profiling hooks in place to track and profile a set of events that we believe provide good insight into how your model is performing in ExecuTorch. Once you have onboarded onto ExecuTorch all you have to do is do a test run with profiling enabled and pass in the profiling buffer dump to the post-procesing tool to get this data.

The events that are tracked right out of the box in the core runtime are:
- Model de-serialization time
- Model loading time
- Inference loop execution time
- Individual operator execution time
- Framework overhead
- Memory consumed (memory allocated through ExecuTorch allocators)

## Running an example with Buck

Here is an example of what a ExecuTorch run + profile + post-procesing workflow looks like.:

This runs the sample program with profiling enabled
```bash
cd executorch
buck2 run -c executorch.prof_enabled=true examples/runtime/portable:executor_runner  -- --model_path add.pte
```
Run the post-processing CLI tool that calls into the same API's listed above and prints out the profiling results in a tabulated format in the terminal.

```bash
cd executorch
buck2 run profiler:profiler_results_cli -- --prof_results_bin prof_result.bin
```

## Runtime Profiling API's

In this section we'll describe the API's that can be used to create new profiling blocks, enable profiling on memory allocators or profile custom events in your application that are outside of the scope of the events already tracked in the core runtime listed above.

`EXECUTORCH_PROFILE_CREATE_BLOCK(name)`

Creates a new profiling block. During post-processing events from all blocks with the same names will be aggregated together. Every block with the same name should have the same number of events, this means that if you have multiple test cases that can take different branches and hence execute different operations due to control flow then each test case that takes a certain branch must be labelled differently.

**Input**:

*name* (const char \*)  - Name of this profiling block

**Returns**:

N/A

#### `EXECUTORCH_DUMP_PROFILE_RESULTS(prof_result)`
After finishing your test run you will need access to the profiling buffer dump which will then be used in the post-processing step to de-serialize and generate the metrics that were tracked in your test run. A call to this API does some background work to serialize the profiling results into a binary dump and updates prof_result with the pointer to this dump and the size of it.

**Input**:

*prof_result* (struct prof_result_t *) - prof_data inside prof_result will be updated to point to the profiling buffer dump and num_bytes inside prof_result will be updated to indicate the size of the profiling buffer dump.

**Returns**:

void

#### `EXECUTORCH_RESET_PROFILE_RESULTS()`

Resets the current profiling block to clear all the events in it.

**Input**:

void

**Returns**:

void

To enable memory profiling all you need to do is call the interface below on the ExecuTorch MemoryAllocator instance that you have created. This will then track all the memory allocations done via this allocator and generate the total size of allocations done after the profiling dump is post-processed.

`enable_profiling(name)`

**Input**:

*name* (const char \*)  - Name of this allocator to tag it with.

**Returns**:

void

## Post-processing profiling API's

Once you have completed your test run and obtained the profiling buffer dump users can use the following Python API's to deserialize the profiling data, aggregate metrics and also print out the results in a pretty table format in the terminal.

#### `deserialize_profile_results(buff: bytes, time_scale: TimeScale = TimeScale.TIME_IN_NS)`

**Input**:

*buff : bytes* - Bytes read from the profiling buffer dump

*time_scale : TimeScale* - Time scale of the profiling data that was collected from the target platform. Refer to the [source code](https://fburl.com/code/zygsx318) for the list of valid values.

**Returns**:

*result : Tuple[Dict[str, List[ProfileEvent]], Dict[str, List[MemEvent]]]* The result returned is a tuple of dictionaries. The first dictionary maps the block name to the corresponding list of profiling events that were aggregated from that block. The second dictionary maps the block name to the corresponding list of memory allocation events that were aggregated from that block.

#### `deserialize_profile_results_files(profile_results_path: str, model_ff_path: str, time_scale: TimeScale = TimeScale.TIME_IN_NS)`:

In this API you pass in the path to the profiling buffer dump and the path to the model flatbuffer file along with the time scale. If this model was emitted and subsquently serialized into a flatbuffer with stacktraces enabled then during the post-processing we'll map the operators that were executed in the model to the corresponding line of python code that this operator maps back to.

*profile_results_path : str* - Path to the profiling buffer dump

*model_ff_path : str* - Path to the model flatbuffer file

*time_scale: TimeScale* - Time scale of the profiling data that was collected from the target platform. Refer to the [source code](https://fburl.com/code/zygsx318) for the list of valid values.

**Returns**:

*result : Tuple[Dict[str, List[ProfileEvent]], Dict[str, List[MemEvent]]]* The result returned is a tuple of dictionaries. The first dictionary maps the block name to the corresponding list of profiling events that were aggregated from that block. The second dictionary maps the block name to the corresponding list of memory allocation events that were aggregated from that block.

#### `profile_aggregate_framework_tax(prof_data: Dict[str, List[ProfileEvent]])`
Through this interface users will be able to generate metrics about the framework overhead that was incurred while executing this model.

**Input**:

*prof_data: Dict[str, List[ProfileEvent]]* - The profiling events dictionary that was returned by the deserialization of the profiling results.

**Returns**:

*result : Dict[str, ProfileEventFrameworkTax]* - A dictionary is returned that maps the block name to the profiling overhead metrics parsed from the profiling data of this block. If this block had no operators executed in it then this will not generate any useful data.

#### `profile_table(profile_data: Dict[str, List[ProfileEvent]], model_buffer=None)`

This will return a list of instances of `PrettyTable`'s on which `print()` can be called to print out to the terminal the data in a tabulated format. Each table in the list contains all the profiling events that were aggregated for blocks with the same names.

**Input**:

*profile_data: Dict[str, List[ProfileEvent]]* - The profiling events dictionary that was returned after the deserialization of the profiling results.

**Returns**:

*table: List[PrettyTable]* - A list of `PrettyTable`'s where each table contains all the profiling events that were aggregated for blocks with the same names.


#### `profile_framework_tax_table(prof_framework_tax_data: Dict[str, ProfileEventFrameworkTax]):`

This will return a list of instances of `PrettyTable`'s on which `print()` can be called to print out to the terminal the data in a tabulated format. Each table in the list contains the framework overhead metrics that were aggregated for blocks with the same names.

**Input**:

*prof_framework_tax_data: Dict[str, ProfileEventFrameworkTax]* - The profiling framework overhead dictionary that was returned by `profile_aggregate_framework_tax`.

*table: List[PrettyTable]* - A list of `PrettyTable`'s where each table contains all the framework overhead metrics that were aggregated for blocks with the same names.

#### `mem_profile_table(mem_allocations: Dict[str, List[MemEvent]])`:

This will return a list of instances of `PrettyTable`'s on which `print()` can be called to print out to the terminal the data in a tabulated format. Each table in the list contains the total memory allocations done from the memory allocators aggregated for blocks with the same names.

**Input**:

*mem_allocations: Dict[str, List[MemEvent]]* - The memory allocations dictionary that was returned after the deserialization of the profiling results.
