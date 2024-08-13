# Check if the PYTHON_EXECUTABLE environment variable is set
if ([string]::IsNullOrEmpty($env:PYTHON_EXECUTABLE)) {
    # Check if the CONDA_DEFAULT_ENV environment variable is empty, equal to "base", or if the "python" command is not executable
    if ([string]::IsNullOrEmpty($env:CONDA_DEFAULT_ENV) -or $env:CONDA_DEFAULT_ENV -eq "base" -or !(Get-Command "python" -ErrorAction SilentlyContinue)) {
        # Set PYTHON_EXECUTABLE to "python3"
        $env:PYTHON_EXECUTABLE = "python3"
    }
    else {
        # Set PYTHON_EXECUTABLE to "python"
        $env:PYTHON_EXECUTABLE = "python"
    }
}

# Set the PIP_EXECUTABLE based on the value of PYTHON_EXECUTABLE
if ($env:PYTHON_EXECUTABLE -eq "python") {
    $env:PIP_EXECUTABLE = "pip"
}
else {
    $env:PIP_EXECUTABLE = "pip3"
}

# Parse options
$EXECUTORCH_BUILD_PYBIND = "OFF"

foreach ($arg in $args) {
    switch ($arg) {
        "--debug" {
            $ENV:DEBUG = "1"
        }
        "--pybind" {
            $EXECUTORCH_BUILD_PYBIND = "ON"
        }
        "coreml" {
            if ($EXECUTORCH_BUILD_PYBIND -eq "ON") {
                $arg_upper = $arg.ToUpper()
                $CMAKE_ARGS = "$CMAKE_ARGS -DEXECUTORCH_BUILD_${arg_upper}=ON"
            }
            else {
                Write-Error "Error: $arg must follow --pybind"
                exit 1
            }
        }
        "mps" {
            if ($EXECUTORCH_BUILD_PYBIND -eq "ON") {
                $arg_upper = $arg.ToUpper()
                $CMAKE_ARGS = "$CMAKE_ARGS -DEXECUTORCH_BUILD_${arg_upper}=ON"
            }
            else {
                Write-Error "Error: $arg must follow --pybind"
                exit 1
            }
        }
        "xnnpack" {
            if ($EXECUTORCH_BUILD_PYBIND -eq "ON") {
                $arg_upper = $arg.ToUpper()
                $CMAKE_ARGS = "$CMAKE_ARGS -DEXECUTORCH_BUILD_${arg_upper}=ON"
            }
            else {
                Write-Error "Error: $arg must follow --pybind"
                exit 1
            }
        }
        default {
            Write-Error "Error: Unknown option $arg"
            exit 1
        }
    }
}

#TODO: support ninja
$CMAKE_ARGS = "$CMAKE_ARGS -T ClangCL"

$NIGHTLY_VERSION = "dev20240716"

# The pip repository that hosts nightly torch packages.
$TORCH_URL = "https://download.pytorch.org/whl/test/cpu"
$TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
$EXIR_REQUIREMENTS = @(
  "torch==2.5.0.${NIGHTLY_VERSION}"
  "torchvision==0.20.0.${NIGHTLY_VERSION}"  # For testing.
)

# pip packages needed for development.
$DEVEL_REQUIREMENTS = @(
  "cmake"  # For building binary targets.
  "pip>=23" # For building the pip package.
  "pyyaml"  # Imported by the kernel codegen tools.
  "setuptools>=63"  # For building the pip package.
  "tomli" # Imported by extract_sources.py when using python < 3.11.
  "wheel"  # For building the pip package archive.
  "zstd"  # Imported by resolve_buck.py.
)

# pip packages needed to run examples.
# TODO(dbort): Make each example publish its own requirements.txt
$EXAMPLES_REQUIREMENTS = @(
  "timm==1.0.7"
  "torchaudio==2.4.0.${NIGHTLY_VERSION}"
  "torchsr==1.0.4"
  "transformers==4.42.4"
)

# Assemble the list of requirements to actually install.
# TODO(dbort): Add options for reducing the number of requirements.
$REQUIREMENTS_TO_INSTALL = @(
    $EXIR_REQUIREMENTS
    $DEVEL_REQUIREMENTS
    $EXAMPLES_REQUIREMENTS
)

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
& $ENV:PIP_EXECUTABLE install --extra-index-url $TORCH_NIGHTLY_URL $REQUIREMENTS_TO_INSTALL

#
# Install executorch pip package. This also makes `flatc` available on the path.
# The --extra-index-url may be necessary if pyproject.toml has a dependency on a
# pre-release or nightly version of a torch package.
#
$ENV:EXECUTORCH_BUILD_PYBIND = $EXECUTORCH_BUILD_PYBIND
$ENV:CMAKE_ARGS = $CMAKE_ARGS
$ENV:CMAKE_BUILD_ARGS = $CMAKE_BUILD_ARGS
& $ENV:PIP_EXECUTABLE install . --no-build-isolation -v --extra-index-url $TORCH_NIGHTLY_URL
