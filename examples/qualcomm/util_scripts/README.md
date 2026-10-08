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

  with open(f"{ws}/input_list.txt", 'w') as f:
      f.write(input_list)
  ```

* Quantize
  ```bash 
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli quantize -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli quantize -a cli_example/simple_model.pt2 -o cli_example/quantize_output -c use_8a8w -i cli_example/input_list.txt --per_channel -m SM8750 --backend htp
  ```
  To quantize for an HTP Flexible Context Binary, pass every target SoC:
  ```bash
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli quantize -a cli_example/simple_model.pt2 -o cli_example/quantize_output -c use_8a8w -i cli_example/input_list.txt --per_channel -m SM8650 SM8750 --backend htp
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
  To generate one HTP FCB containing multiple target contexts:
  ```bash
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -a cli_example/quantize_output/simple_model_quantized.pt2 -o cli_example/compile_output -m SM8650 SM8750 --backend htp
  ```
* (Optional) Compile pre-generated context binary to .pte program
  ```bash
  # `pip install pydot` if package is missing
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli compile -a model.bin -o path/to/model/output -m SM8750 --backend htp
  ```
* Artifacts for .pte file and figure of graph information
  - `cli_example/compile_output/simple_model_quantized.pte`
  - `cli_example/compile_output/simple_model_quantized.svg`

### Executing Program

* Execute .pte program
  ```bash
  # user could get more information via: PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli execute -h
  PYTHONPATH=.. python -m examples.qualcomm.util_scripts.cli execute -a cli_example/compile_output/simple_model_quantized.pte -o cli_example/execute_output -i cli_example/input_list.txt -H $HOST_NAME -s $DEVICE_SERIAL -b build-android -m SM8750 --backend htp
  ```
* Artifacts for .pte file and figure of graph information
  - `cli_example/execute_output/output_{data_index}_{output_index}.pt`.<br/>
  `data_index` represents the sequence of dataset, `output_index` stands for the order of graph output.

## HTP Profiling Examples

These examples demonstrate Qualcomm HTP profiling flows:

* `htp_profiling_on_device_op_trace_online.py`: on-device OpTrace with `online_prepare=True`.
* `htp_profiling_on_device_op_trace_offline.py`: on-device OpTrace with `online_prepare=False` and `profile_level=3`.
* `htp_profiling_on_host_hextimate.py`: host-side Hextimate.

## LPAI Depth-First Cut (DFC)

Depth-first cut is a **model-build-time, memory-fitting** mechanism for the LPAI backend. It
splits one op's compute (and its working set) into smaller spatial tiles so a graph that
otherwise wouldn't fit into LPAI's on-chip memory can still be compiled. It is not a speedup
lever — on a graph that already compiles fine without it, expect comparable or slightly worse
latency.

Requirements and limitations:
* QNN SDK >= 2.51
* Support is limited to specific operations; refer to the qnn sdk document’s lpai backend section for details.
* A cut entry naming more than one node must enumerate every node on every path from the first to the last listed node.
* If the node you want to cut is normally fused with a following node by the backend (e.g. conv +
  activation), include that node in the cut entry too. Cutting only the conv breaks the fusion and
  can cause an accuracy drop. Check the QNN SDK documentation for which op patterns the backend
  fuses (super groups) before deciding what to include in a cut entry.
* Only supports a **fully-delegated** graph (the whole model lowers to a single QNN partition). If the graph is split into multiple partitions, a partition that doesn't contain the named node(s) may fail to
  compile. The partitioner logs a warning when it detects this combination, but does not
  block the export.
* If multiple methods in the same process request different cut plans, only the first one actually takes effect — later ones are silently
  ignored (a warning is logged). This is a known limitation of the current multi-method design.

Since writing a cut plan requires knowing the exact node names in the QNN graph, use
`--dump_qnn_graph_path` to dump the graph first and inspect it before deciding on a cut plan.

### Cut plan schema

The feature is configured through two JSON files, threaded in via
`generate_lpai_compiler_spec(lpai_custom_prepare_config_file=...)`
(`backends/qualcomm/utils/utils.py`):

* `custom_prepare_config.json` — manifest passed to the LPAI backend. Currently the only
  supported key is `cut_plan`, pointing at the cut plan file:
  ```json
  {"cut_plan": "/path/to/cut_plan_nodes.json"}
  ```
* `cut_plan_nodes.json` — a list of cut entries:
  ```json
  [{"nodes": ["aten_convolution_default"], "h_branches": 2, "w_branches": 1}]
  ```

  | Field | Type | Description |
  |---|---|---|
  | `nodes` | array of strings | Exported-graph node name(s) forming the cut group (e.g. `aten_convolution_default`, `aten_convolution_default_1`, ...). For more than one node, every node on every path from the first to the last listed node must be included. |
  | `h_branches` | integer | Number of tiles along the spatial height dimension. |
  | `w_branches` | integer | Number of tiles along the spatial width dimension. |

### Running the example

`lpai_depth_first_cut.py` builds a tiny single-conv model, writes both JSON files, then lowers
and runs it on-device once with the cut plan applied, and compares the output against the CPU
reference:

```bash
PYTHONPATH=.. python -m examples.qualcomm.util_scripts.lpai_depth_first_cut \
    --host $HOST_NAME --device $DEVICE_SERIAL --soc_model SM8850 --build_folder build-android \
    --backend lpai -a cli_example/lpai_dfcut_output \
    --nodes aten_convolution_default --h_branches 2 --w_branches 1
```

* Artifacts for the generated cut plan and compiled program:
  - `cli_example/lpai_dfcut_output/cut_plan_nodes.json`
  - `cli_example/lpai_dfcut_output/custom_prepare_config.json`
  - `cli_example/lpai_dfcut_output/lpai_dfcut.pte`

# Generate ET Record
This section describes how to generate an ET record for a .pte program using the provided script.
  * Generate ET record for .pte using the provided script:
    ```bash
    # Example usage to generate ET record and inspect execution statistics
    PYTHONPATH=.. python -m examples.qualcomm.util_scripts.gen_etrecord \
      --build_folder build-android \
      --device $DEVICE_SERIAL \
      --soc_model SM8750 \
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
