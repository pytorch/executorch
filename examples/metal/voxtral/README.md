# Voxtral Metal Backend Example

This directory contains and end-to-end example for running Voxtral on ExecuTorch Metal backend.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Conda environment manager

## Voxtral Example

The Voxtral example demonstrates how to:
1. Set up a Python environment
2. Export the Mistral Voxtral model to ExecuTorch format
3. Build the Voxtral Metal runner
4. Run inference on audio input

**Required Arguments:**

- `<env_name>` - Name of the conda environment to create (e.g., `voxtral-example`)
- `<audio_path>` - Path to your audio file for inference (e.g., `/path/to/audio.wav`)
- `[export_dir]` - (Optional) Directory to store exported model files (default: `voxtral`)

**Example:**

```bash
examples/metal/voxtral/e2e.sh \
  --env-name voxtral-example \
  --create-env --setup-env \
  --export --export-dir ~/Desktop/voxtral \
  --build \
  --run --audio-path ~/Desktop/audio.wav
```

This will automatically:
1. Create a conda environment named `voxtral-example`
2. Install all dependencies
3. Export the Voxtral model to the `~/Desktop/voxtral` directory
4. Build the Voxtral Metal runner
5. Run inference on `~/Desktop/voxtral/audio.wav`

### Step-by-Step Manual Process

If you prefer to run each step individually, follow these steps:

#### Step 1: Set up the Python Environment

```bash
conda create -yn <env_name> python=3.11
conda activate <env_name>
./examples/metal/setup_python_env.sh
```

**Arguments:**
- `<env_name>` - Name of the conda environment (e.g., `voxtral-example`)

This will:
- Create a new conda environment
- Install all required dependencies including ExecuTorch and Optimum-ExecuTorch

#### Step 2: Export the Model

```bash
./examples/metal/voxtral/export.sh [export_dir]
```

**Arguments:**
- `[export_dir]` - (Optional) Directory to store exported model files (default: `voxtral`)

**Example:**

```bash
./examples/metal/voxtral/export.sh ~/Desktop/voxtral
```

This will:
- Download the Mistral Voxtral model
- Export it to ExecuTorch format with Metal optimizations
- Save model files (`.pte`), metadata, and preprocessor to the specified directory

#### Step 3: Build the Voxtral Metal Runner

```bash
./examples/metal/voxtral/build.sh
```

#### Step 4: Run Inference

```bash
./examples/metal/voxtral/run.sh <audio_path> [export_dir]
```

**Required Arguments:**
- `<audio_path>` - Path to your audio file (e.g., `/path/to/audio.wav`)

**Optional Arguments:**
- `[export_dir]` - Directory containing exported model files (default: `voxtral`)

**Example:**

```bash
./examples/metal/voxtral/run.sh ~/Desktop/audio.wav ~/Desktop/voxtral
```

This will:
- Validate that all required model files exist
- Load the model and preprocessor
- Run inference on the provided audio
- Display timing information
