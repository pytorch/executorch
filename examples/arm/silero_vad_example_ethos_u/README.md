# Silero VAD Ethos-U Example

Run Silero voice activity detection on Corstone-320 / Ethos-U85, then compare
its speech probabilities with the host reference.

## Set Up

Run all commands from the ExecuTorch repository root. Use Python 3.12 with
development headers and `venv`, plus the
[host build prerequisites](../../../docs/source/using-executorch-building-from-source.md#software-requirements)
on Linux or Apple Silicon macOS 14+.

<details>
<summary>macOS: install the FVP wrappers first</summary>

Start Docker, then install [FVPs-on-Mac](https://github.com/Arm-Examples/FVPs-on-Mac)
for FVP 11.31.28:

```bash
mkdir -p examples/arm/arm-scratch "$HOME/.armlm"
git clone https://github.com/Arm-Examples/FVPs-on-Mac.git \
  examples/arm/arm-scratch/FVPs-on-Mac
git -C examples/arm/arm-scratch/FVPs-on-Mac checkout --detach \
  cc7a4be5456db085619647811fac29e9795f806e
bash examples/arm/arm-scratch/FVPs-on-Mac/build.sh
export PATH="$(pwd)/examples/arm/arm-scratch/FVPs-on-Mac/bin:$PATH"
export FVP_MOUNT_DIR="$(pwd)"
FVP_Corstone_SSE-320 --version
```

</details>

Create a Python environment and install the released Python packages and Arm tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
git submodule update --init --recursive
python -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
./examples/arm/setup.sh --i-agree-to-the-contained-eula \
  --disable-ethos-u-deps --enable-baremetal-toolchain --enable-fvps
python -m pip install "executorch[ethos_u]==1.5.1" "torchao==0.18.0" "cmake<4" pytest
source examples/arm/arm-scratch/setup_path.sh
```

The setup command accepts the Arm tool licenses. If reusing an environment
installed for this checkout, activate it instead of creating `.venv`.

In each new shell, restore the environment and tool paths:

```bash
source .venv/bin/activate
source examples/arm/arm-scratch/setup_path.sh
if [ "$(uname -s)" = Darwin ]; then
  export PATH="$(pwd)/examples/arm/arm-scratch/FVPs-on-Mac/bin:$PATH"
  export FVP_MOUNT_DIR="$(pwd)"
fi
```

## Run the Example

1. [Prepare inputs and export](model_export/README.md).
2. [Build, run, and validate](runtime/README.md).
