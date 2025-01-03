#!/bin/bash

# Function to download files from the Hugging Face Hub
# Arguments:
# 1. model_id: The Hugging Face repository ID (e.g., "organization/model_name")
# 2. subdir: The optional subdirectory in the repo to look for files (pass "" if not used)
# 3. file_names: A space-separated list of filenames to be downloaded
# Returns:
# The directory containing the downloaded files
function download_hf_files() {
  local model_id="$1"
  local subdir="$2"
  shift 2
  local file_names=("$@")  # Capture all remaining arguments as an array

  local download_dir

  # Use the first file to determine the download directory
  download_dir=$(python3 -c "
from huggingface_hub import hf_hub_download
# Download the first file and get its directory
path = hf_hub_download(
    repo_id='${model_id}',
    filename='${subdir:+${subdir}/}${file_names[0]}'
)
import os
print(os.path.dirname(path))")

  if [ $? -ne 0 ]; then
    echo "Error: Failed to determine download directory from ${file_names[0]}" >&2
    return 1
  fi

  # Download remaining files into the same directory
  for file_name in "${file_names[@]:1}"; do
    python3 -c "
from huggingface_hub import hf_hub_download
# Download the file
hf_hub_download(
    repo_id='${model_id}',
    filename='${subdir:+${subdir}/}${file_name}'
)"

    if [ $? -ne 0 ]; then
      echo "Error: Failed to download ${file_name} from ${model_id}" >&2
      return 1
    fi
  done

  # Return the directory containing the downloaded files
  echo "$download_dir"
}

# Check if script is called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Parse arguments from CLI
  while [[ $# -gt 0 ]]; do
    case $1 in
      --model_id)
        MODEL_ID="$2"
        shift 2
        ;;
      --subdir)
        SUBDIR="$2"
        shift 2
        ;;
      --files)
        shift
        FILES_TO_DOWNLOAD=()
        while [[ $# -gt 0 && $1 != --* ]]; do
          FILES_TO_DOWNLOAD+=("$1")
          shift
        done
        ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
  done

  # Validate required arguments
  if [ -z "$MODEL_ID" ] || [ ${#FILES_TO_DOWNLOAD[@]} -eq 0 ]; then
    echo "Usage: $0 --model_id <model_id> --subdir <subdir> --files <file1> [<file2> ...]" >&2
    exit 1
  fi

  # Call the function
  DOWNLOAD_DIR=$(download_hf_files "$MODEL_ID" "$SUBDIR" "${FILES_TO_DOWNLOAD[@]}")
  if [ $? -eq 0 ]; then
    echo "$DOWNLOAD_DIR"
  else
    exit 1
  fi
fi
