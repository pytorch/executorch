#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Initialize variables
CREATE_ENV=false
SETUP_ENV=false
EXPORT=false
BUILD=false
RUN=false
EXPORT_DIR="voxtral"
AUDIO_PATH=""
ENV_NAME=""

# Function to display usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --env-name NAME        Name of the conda environment (required)"
  echo "  --create-env           Create the Python environment"
  echo "  --setup-env            Set up the Python environment"
  echo "  --export-dir DIR       Specify the export directory (default: voxtral)"
  echo "  --export               Export the Voxtral model"
  echo "  --build                Build the Voxtral runner"
  echo "  --audio-path PATH      Path to the input audio file"
  echo "  --run                  Run the Voxtral model"
  echo "  -h, --help             Display this help message"
  echo ""
  echo "Example:"
  echo "  $0 --env-name metal-backend --setup-env --export --build --audio-path audio.wav --run"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --create-env)
      CREATE_ENV=true
      shift
      ;;
    --setup-env)
      SETUP_ENV=true
      shift
      ;;
    --export-dir)
      EXPORT_DIR="$2"
      shift 2
      ;;
    --export)
      EXPORT=true
      shift
      ;;
    --build)
      BUILD=true
      shift
      ;;
    --audio-path)
      AUDIO_PATH="$2"
      shift 2
      ;;
    --run)
      RUN=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required options
if [ -z "$ENV_NAME" ]; then
  echo "Error: --env-name is required"
  exit 1
fi

if [ "$RUN" = true ]; then
  if [ -z "$AUDIO_PATH" ]; then
    echo "Error: --audio-path is required when using --run"
    exit 1
  fi
fi

# Execute create-env
if [ "$CREATE_ENV" = true ]; then
  echo "Creating the Python environment $ENV_NAME ..."
  conda create -yn "$ENV_NAME" python=3.11
fi

# Execute setup-env
if [ "$SETUP_ENV" = true ]; then
  echo "Setting up Python environment $ENV_NAME ..."
  conda run -n "$ENV_NAME" ./examples/metal/setup_python_env.sh
fi

# Execute export
if [ "$EXPORT" = true ]; then
  echo "Exporting Voxtral model to $EXPORT_DIR ..."
  conda run -n "$ENV_NAME" ./examples/metal/voxtral/export.sh "$EXPORT_DIR"
fi

# Execute build
if [ "$BUILD" = true ]; then
  echo "Building Voxtral runner ..."
  conda run -n "$ENV_NAME" ./examples/metal/voxtral/build.sh
fi

# Execute run
if [ "$RUN" = true ]; then
  echo "Running Voxtral with audio: $AUDIO_PATH"
  ./examples/metal/voxtral/run.sh "$AUDIO_PATH" "$EXPORT_DIR"
fi

echo "Done!"
