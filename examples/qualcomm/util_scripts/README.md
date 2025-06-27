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
