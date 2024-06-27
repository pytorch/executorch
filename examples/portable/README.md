# ExecuTorch in Portable Mode

This dir contains demos to illustrate an end-to-end workflow of using ExecuTorch in [portable mode](../../docs/source/concepts.md#portable-mode-lean-mode).


## Directory structure
```bash
examples/portable
├── scripts                           # Python scripts to illustrate export workflow
│   ├── export.py
│   └── export_and_delegate.py
├── custom_ops                        # Contains examples to register custom operators into PyTorch as well as register its kernels into ExecuTorch runtime
├── executor_runner                   # Contains an example C++ wrapper around the ExecuTorch runtime
└── README.md                         # This file
```

## Using portable mode

We will walk through an example model to generate a `.pte` file in [portable mode](../../docs/source/concepts.md#portable-mode-lean-mode) from a python `torch.nn.module`
from the [`models/`](../models) directory using scripts in the `portable/scripts` directory. Then we will run on the `.pte` model on the ExecuTorch runtime. For that we will use `executor_runner`.


1. Following the setup guide in [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup)
you should be able to get the basic development environment for ExecuTorch working.

2. Using the script `portable/scripts/export.py` generate a model binary file by selecting a
model name from the list of available models in the `models` dir.


```bash
cd executorch # To the top level dir

# To get a list of example models
python3 -m examples.portable.scripts.export -h

# To generate a specific pte model
python3 -m examples.portable.scripts.export --model_name="mv2" # for MobileNetv2

# This should generate ./mv2.pte file, if successful.
```

Use `-h` (or `--help`) to see all the supported models.

3. Once we have the model binary (`.pte`) file, then let's run it with ExecuTorch runtime using the `executor_runner`.

```bash
# Build the tool from the top-level `executorch` directory.
(rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DEXECUTORCH_PAL_DEFAULT=posix ..) \
  && cmake --build cmake-out -j32 --target executor_runner

# Run the tool on the generated model.
./cmake-out/executor_runner --model_path mv2.pte
```

This will run the model with all input tensor elements set to `1`, and print
the outputs. For example:
```
I 00:00:00.004885 executorch:executor_runner.cpp:73] Model file mv2.pte is loaded.
I 00:00:00.004902 executorch:executor_runner.cpp:82] Using method forward
I 00:00:00.004906 executorch:executor_runner.cpp:129] Setting up planned buffer 0, size 18652672.
I 00:00:00.007243 executorch:executor_runner.cpp:152] Method loaded.
I 00:00:00.007490 executorch:executor_runner.cpp:162] Inputs prepared.
I 00:00:06.887939 executorch:executor_runner.cpp:171] Model executed successfully.
I 00:00:06.887975 executorch:executor_runner.cpp:175] 1 outputs:
Output 0: tensor(sizes=[1, 1000], [
  -0.50986, 0.300638, 0.0953877, 0.147722, 0.231202, 0.338555, 0.20689, -0.057578, -0.389269, -0.060687,
  -0.0213992, -0.121035, -0.288955, 0.134054, -0.171976, -0.0603627, 0.0203591, -0.0585333, 0.337855, -0.0718644,
  0.490758, 0.524144, 0.197857, 0.122066, -0.35913, 0.109461, 0.347747, 0.478515, 0.226558, 0.0363523,
  ...,
  -0.227163, 0.567008, 0.202894, 0.71008, 0.421649, -0.00655106, 0.0114818, 0.398908, 0.0349851, -0.163213,
  0.187843, -0.154387, -0.22716, 0.150879, 0.265103, 0.087489, -0.188225, 0.0213046, -0.0293779, -0.27963,
  0.421221, 0.10045, -0.506771, -0.115818, -0.693015, -0.183256, 0.154783, -0.410679, 0.0119293, 0.449714,
])
```

## Custom Operator Registration

Explore the demos in the [`custom_ops/`](./custom_ops) directory to learn how to register custom operators into ExecuTorch as well as register its kernels into ExecuTorch runtime.
