# CLI Tool for Quantize / Compile / Deploy PyTorch Model with QNN Backend

An easy-to-use tool for quantizing / compiling / executing .pte program with Qualcomm AI Engine Direct. Tool is verified with [host environement](../../../docs/source/backends-qualcomm.md#host-os).

## Description

This tool aims for users who want to deploy models with ExecuTorch runtime. It's possible for them to produce .pte program in few steps.<br/>

### Quantizing Model

* Save torch.nn.Module with .pt2 format & prepare input data
  ```bash
  # create workspace for following operations
  cd path/to/executorch
  mkdir cli_example
  ```
  ```python
  # take SimpleModel as an example
  import torch
  from executorch.backends.qualcomm.tests.models import SimpleModel
  from pathlib import Path
  # make example inputs
  example_inputs = (torch.randn(1, 32, 28, 28), torch.randn(1, 32, 28, 28))
  # generate ExportedProgram
  ep = torch.export.export(SimpleModel(), example_inputs)
  # save to workspace
  ws = f"{Path().cwd()}/cli_example"
  torch.export.save(ep, f"{ws}/simple_model.pt2")
  # prepare calibration dataset: 2 sets of data with 2 inputs each
  input_list = ""
  for i in range(2):
      current_input = ""
      for j in range(2):
          file_name = f"{ws}/input_{i}_{j}.pt"
          torch.save(torch.randn(1, 32, 28, 28), file_name)
          current_input += f"{file_name} "
      input_list += f"{current_input.strip()}\n"

  with open(f"{ws}/input_list", 'w') as f:
      f.write(input_list)
  ```

* Quantize
  ```bash 
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli quantize -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli quantize -a cli_example/simple_model.pt2 -o cli_example/quantize_output -c use_8a8w -i cli_example/input_list --per_channel
  ```
* Artifacts for quantized .pt2 file
  - `cli_example/quantize_output/simple_model_quantized.pt2`


### Compiling Program

* Compile .pt2 to .pte program
  ```bash
  # `pip install pydot` if package is missing
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -a cli_example/quantize_output/simple_model_quantized.pt2 -o cli_example/compile_output -m SM8750
  ```
* (Optional) Compile pre-generated context binary to .pte program
  ```bash
  # `pip install pydot` if package is missing
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -a model.bin -o path/to/model/output -m SM8750
  ```
* Artifacts for .pte file and figure of graph information
  - `cli_example/compile_output/simple_model_quantized.pte`
  - `cli_example/compile_output/simple_model_quantized.svg`

### Executing Program

* Execute .pte program
  ```bash
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli execute -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli execute -a cli_example/compile_output/simple_model_quantized.pte -o cli_example/execute_output -i cli_example/input_list -s $DEVICE_SERIAL -b build-android -m SM8750
  ```
* Artifacts for .pte file and figure of graph information
  - `cli_example/execute_output/output_{data_index}_{output_index}.pt`.<br/>
  `data_index` represents the sequence of dataset, `output_index` stands for the order of graph output.

# Generate ET Record
This section describes how to generate an ET record for a .pte program using the provided script.
  * Generate ET record for .pte using the provided script:
    ```bash
    # Example usage to generate ET record and inspect execution statistics
    PYTHONPATH=.. python -m examples.qualcomm.util_scripts.gen_etrecord \
      -b build-android \
      --device $DEVICE_SERIAL \
      --model SM8750 \
    ```
  * This script will:
    - Quantize and compile a sample model to generate `.pte` file.
    - Push the model and input data to the device and execute the program.
    - Retrieve the execution dump from the device and generate an ET record (`etrecord.bin`).
    - Use the Inspector API to display execution statistics.

  * Artifacts generated:
    - `qnn_simple_model.pte`: Compiled program.
    - `etdump.etdp`: Execution dump from device.
    - `etrecord.bin`: ET record for analysis.
    - Printed statistics table in the console.

  * refer to the [runtime-profiling](https://docs.pytorch.org/executorch/stable/runtime-profiling.html) for more details.

## Example console output:
| event_block_name | event_name                                      | raw       | p10 (cycles) | p50 (cycles) | p90 (cycles) | avg (cycles) | min (cycles) | max (cycles) | op_types | delegate_debug_identifier             | stack_traces | module_hierarchy | is_delegated_op | delegate_backend_name | debug_data | start_time |
|------------------|--------------------------------------------------|-----------|--------------|--------------|--------------|---------------|---------------|---------------|----------|----------------------------------------|---------------|------------------|------------------|------------------------|------------|-------------|
| ...              | ...                                  | ...           | ...                    |        |
| Execute          | aten_relu_default_3:OpId_60 (cycles)            | [2045.0]  | 2045.0       | 2045.0       | 2045.0       | 2045.0        | 2045.0        | 2045.0        | []       | aten_relu_default_3:OpId_60 (cycles)         | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_add_tensor:OpId_61 (cycles)                | [10271.0] | 10271.0      | 10271.0      | 10271.0      | 10271.0       | 10271.0       | 10271.0       | []       | aten_add_tensor:OpId_61 (cycles)             | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_permute_copy_default_4:OpId_63 (cycles)    | [31959.0] | 31959.0      | 31959.0      | 31959.0      | 31959.0       | 31959.0       | 31959.0       | []       | aten_permute_copy_default_4:OpId_63 (cycles) | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_mean_dim:OpId_65 (cycles)                  | [11008.0] | 11008.0      | 11008.0      | 11008.0      | 11008.0       | 11008.0       | 11008.0       | []       | aten_mean_dim:OpId_65 (cycles)               | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_view_copy_default:OpId_67 (cycles)         | [5893.0]  | 5893.0       | 5893.0       | 5893.0       | 5893.0        | 5893.0        | 5893.0        | []       | aten_view_copy_default:OpId_67 (cycles)      | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_linear_default:OpId_70 (cycles)            | [0.0]     | 0.0          | 0.0          | 0.0          | 0.0           | 0.0           | 0.0           | []       | aten_linear_default:OpId_70 (cycles)         | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| Execute          | aten_hardtanh_default:OpId_72 (cycles)          | [9799.0]  | 9799.0       | 9799.0       | 9799.0       | 9799.0        | 9799.0        | 9799.0        | []       | aten_hardtanh_default:OpId_72 (cycles)       | {}        | {}               | True             | QnnBackend             | []         | [0]         |
| ...              | ...                                  | ...        | ...                    |
