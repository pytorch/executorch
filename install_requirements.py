import os
import platform
import re
import sys
import subprocess

# Before doing anything, cd to the directory containing this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Find the names of the python tools to use.
PYTHON_EXECUTABLE = os.getenv('PYTHON_EXECUTABLE')

if not PYTHON_EXECUTABLE:
    CONDA_DEFAULT_ENV = os.getenv('CONDA_DEFAULT_ENV')
    if not CONDA_DEFAULT_ENV or CONDA_DEFAULT_ENV == "base" or not subprocess.call(["which", "python"]):
        PYTHON_EXECUTABLE = "python3"
    else:
        PYTHON_EXECUTABLE = "python"

if PYTHON_EXECUTABLE == "python":
    PIP_EXECUTABLE = "pip"
else:
    PIP_EXECUTABLE = "pip3"

print(f"Using Python executable: {PYTHON_EXECUTABLE}")
print(f"Using Pip executable: {PIP_EXECUTABLE}")

def python_is_compatible():
    # Scrape the version range from pyproject.toml, which should be in the current directory.
    version_specifier = None
    with open('pyproject.toml', 'r') as file:
        for line in file:
            if line.startswith('requires-python'):
                match = re.search(r'"([^"]*)"', line)
                if match:
                    version_specifier = match.group(1)
                    break

    if not version_specifier:
        print("WARNING: Skipping python version check: version range not found", file=sys.stderr)
        return False

    # Install the packaging module if necessary.
    try:
        import packaging
    except ImportError:
        subprocess.check_call([PIP_EXECUTABLE, 'install', 'packaging'])
    # Compare the current python version to the range in version_specifier. Exits
    # with status 1 if the version is not compatible, or with status 0 if the
    # version is compatible or the logic itself fails.
    try:
        import packaging.version
        import packaging.specifiers

        python_version = packaging.version.parse(platform.python_version())
        version_range = packaging.specifiers.SpecifierSet(version_specifier)
        if python_version not in version_range:
            print(
                f"ERROR: ExecuTorch does not support python version {python_version}: must satisfy \"{version_specifier}\"",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        print(f"WARNING: Skipping python version check: {e}", file=sys.stderr)
        sys.exit(0)
        return False
    return True

if not python_is_compatible():
    sys.exit(1)

# Parse options.
EXECUTORCH_BUILD_PYBIND = "OFF"
CMAKE_ARGS = os.getenv("CMAKE_ARGS", "")
CMAKE_BUILD_ARGS = os.getenv("CMAKE_BUILD_ARGS", "")

for arg in sys.argv[1:]:
    if arg == "--pybind":
        EXECUTORCH_BUILD_PYBIND = "ON"
    elif arg in ["coreml", "mps", "xnnpack"]:
        if EXECUTORCH_BUILD_PYBIND == "ON":
            arg_upper = arg.upper()
            CMAKE_ARGS += f" -DEXECUTORCH_BUILD_{arg_upper}=ON"
        else:
            print(f"Error: {arg} must follow --pybind")
            sys.exit(1)
    else:
        print(f"Error: Unknown option {arg}")
        sys.exit(1)

print(f"EXECUTORCH_BUILD_PYBIND: {EXECUTORCH_BUILD_PYBIND}")
print(f"CMAKE_ARGS: {CMAKE_ARGS}")
print(f"CMAKE_BUILD_ARGS: {CMAKE_BUILD_ARGS}")

NIGHTLY_VERSION = "dev20240716"

# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
EXIR_REQUIREMENTS = [
    f"torch==2.5.0.{NIGHTLY_VERSION}",
    f"torchvision==0.20.0.{NIGHTLY_VERSION}"  # For testing.
]

# pip packages needed for development.
DEVEL_REQUIREMENTS = [
    "cmake",  # For building binary targets.
    "pip>=23",  # For building the pip package.
    "pyyaml",  # Imported by the kernel codegen tools.
    "setuptools>=63",  # For building the pip package.
    "tomli",  # Imported by extract_sources.py when using python < 3.11.
    "wheel",  # For building the pip package archive.
    "zstd"  # Imported by resolve_buck.py.
]

# pip packages needed to run examples.
# TODO: Make each example publish its own requirements.txt
EXAMPLES_REQUIREMENTS = [
    "timm==1.0.7",
    f"torchaudio==2.4.0.{NIGHTLY_VERSION}",
    "torchsr==1.0.4",
    "transformers==4.42.4"
]

# Assemble the list of requirements to actually install.
# TODO: Add options for reducing the number of requirements.
REQUIREMENTS_TO_INSTALL = EXIR_REQUIREMENTS + DEVEL_REQUIREMENTS + EXAMPLES_REQUIREMENTS

print("Requirements to install:")
for requirement in REQUIREMENTS_TO_INSTALL:
    print(requirement)

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
subprocess.check_call([PIP_EXECUTABLE, "install", *REQUIREMENTS_TO_INSTALL, "--extra-index-url", TORCH_NIGHTLY_URL])

#
# Install executorch pip package. This also makes `flatc` available on the path.
# The --extra-index-url may be necessary if pyproject.toml has a dependency on a
# pre-release or nightly version of a torch package.
#

# Set environment variables
os.environ["EXECUTORCH_BUILD_PYBIND"] = EXECUTORCH_BUILD_PYBIND
os.environ["CMAKE_ARGS"] = CMAKE_ARGS
os.environ["CMAKE_BUILD_ARGS"] = CMAKE_BUILD_ARGS

# Run the pip install command
subprocess.check_call([
    PIP_EXECUTABLE, "install", ".", "--no-build-isolation", "-v",
    "--extra-index-url", TORCH_NIGHTLY_URL
])
