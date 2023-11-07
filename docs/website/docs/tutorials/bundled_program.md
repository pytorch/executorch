DEPRECATED: This document is moving to //executorch/docs/source/sdk-bundled-io.md

# Bundled Program

## Introduction
Bundled Program is a wrapper around the core ExecuTorch program designed to help users wrapping test cases and other related info with the models they deploy. Bundled Program is not necessarily a core part of the program and not needed for its execution but is more necessary for various other use-cases, especially for model correctness evaluation such as e2e testing during model bring-up etc.

Overall procedure can be broken into two stages, and in each stage we are supporting:
* **Emit stage**: Bundling test I/O cases as well as other useful info in key-value pairs along with the ExecuTorch program.
* **Runtime stage**: Accessing, executing and verifying the bundled test cases during runtime.

## Emit stage

 This stage mainly focuses on the creation of a BundledProgram, and dump it out to the disk as a flatbuffer file. Please refer to Bento notebook [N2744997](https://www.internalfb.com/intern/anp/view/?id=2744997) for details on how to create a bundled program.

## Runtime Stage
This stage mainly focuses on executing the model with the bundled inputs and and comparing the model's output with the bundled expected output. We provide multiple APIs to handle the key parts of it.

### Get executorch program ptr from BundledProgram buffer
We need the pointer to executorch program to do the execution. To unify the process of loading and executing BundledProgram and Program flatbuffer, we create an API:
 ```c++

/**
 * Finds the serialized ExecuTorch program data in the provided file data.
 *
 * The returned buffer is appropriate for constructing a
 * torch::executor::Program.
 *
 * Calling this is only necessary if the file could be a bundled program. If the
 * file will only contain an unwrapped ExecuTorch program, callers can construct
 * torch::executor::Program with file_data directly.
 *
 * @param[in] file_data The contents of an ExecuTorch program or bundled program
 *                      file.
 * @param[in] file_data_len The length of file_data, in bytes.
 * @param[out] out_program_data The serialized Program data, if found.
 * @param[out] out_program_data_len The length of out_program_data, in bytes.
 *
 * @returns Error::Ok if the program was found, and
 *     out_program_data/out_program_data_len point to the data. Other values
 *     on failure.
 */
Error GetProgramData(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len);
```

Here's an example of how to use the GetProgramData API:
```c++
  std::shared_ptr<char> buff_ptr;
  size_t buff_len;

// FILE_PATH here can be either BundledProgram or Program flatbuffer file.
  Error status = torch::executor::util::read_file_content(
      FILE_PATH, &buff_ptr, &buff_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "read_file_content() failed with status 0x%" PRIx32,
      status);

  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");

  const void* program_ptr;
  size_t program_len;
  status = torch::executor::bundled_program::GetProgramData(
      buff_ptr.get(), buff_len, &program_ptr, &program_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "GetProgramData() failed with status 0x%" PRIx32,
      status);
```

### Load bundled input to ExecutionPlan
To execute the program on the bundled input, we need to load the bundled input into the ExecutionPlan. Here we provided an API called `torch::executor::bundled_program::LoadBundledInput`:

```c++

/**
 * Load testset_idx-th bundled input of method_idx-th Method test in
 * bundled_program_ptr to given Method.
 *
 * @param[in] method The Method to verify.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] testset_idx  The index of input needs to be set into given Method.
 *
 * @returns Return Error::Ok if load successfully, or the error happens during
 * execution.
 */
__ET_NODISCARD Error LoadBundledInput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    size_t testset_idx);
```

### Verify the plan's output.
We call `torch::executor::bundled_program::VerifyResultWithBundledExpectedOutput` to verify the method's output with bundled expected outputs. Here's the details of this API:

```c++
/**
 * Compare the Method's output with testset_idx-th bundled expected
 * output in method_idx-th Method test.
 *
 * @param[in] method The Method to extract outputs from.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] testset_idx  The index of expected output needs to be compared.
 * @param[in] rtol Relative tolerance used for data comparsion.
 * @param[in] atol Absolute tolerance used for data comparsion.
 *
 * @returns Return Error::Ok if two outputs match, or the error happens during
 * execution.
 */
__ET_NODISCARD Error VerifyResultWithBundledExpectedOutput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    size_t testset_idx,
    double rtol = 1e-5,
    double atol = 1e-8);

```

### Example

Here we provide an example about how to run the bundled program step by step. Most of the code are borrowed from "fbcode/executorch/sdk/fb/runners/executor_runner.cpp" and please review that file if you need more info and context:

```c++
    // method_name is the name for the method we want to test
    // memory_manager is the executor::MemoryManager variable for executor memory allocation.
    // program is the executorch program.
    Result<Method> method = program->load_method(method_name, &memory_manager);
    EXECUTORCH_END_PROF(prof_tok);
    ET_CHECK_MSG(
        method.ok(),
        "load_method() failed with status 0x%" PRIx32,
        method.error());

    // Load testset_idx-th input in the buffer to plan
    status = torch::executor::bundled_program::LoadBundledInput(
          *method,
          program_data.bundled_program_data(),
          FLAGS_testset_idx);
      ET_CHECK_MSG(
          status == Error::Ok,
          "LoadBundledInput failed with status 0x%" PRIx32,
          status);

    // Execute the plan
    status = method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "method->execute() failed with status 0x%" PRIx32,
        status);

    // Verify the result.
    status = torch::executor::bundled_program::VerifyResultWithBundledExpectedOutput(
          *method,
          program_data.bundled_program_data(),
          FLAGS_testset_idx,
          FLAGS_rtol,
          FLAGS_atol);
      ET_CHECK_MSG(
          status == Error::Ok,
          "Bundle verification failed with status 0x%" PRIx32,
          status);

```
