#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

BUILD_TOOL=$2
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

BACKEND=$3
if [[ -z "${BACKEND:-}" ]]; then
  echo "Missing backend (require portable or xnnpack), exiting..."
  exit 1
fi

UPLOAD_DIR=${4:-}

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

# Just set this variable here, it's cheap even if we use buck2
CMAKE_OUTPUT_DIR=cmake-out
EXPORTED_MODEL=${MODEL_NAME}

prepare_artifacts_upload() {
  if [ -n "$UPLOAD_DIR" ]; then
    echo "Preparing for uploading generated artifacs"
    zip -j model.zip "${EXPORTED_MODEL}"
    mkdir -p "${UPLOAD_DIR}"
    mv model.zip "${UPLOAD_DIR}"
  fi
}


build_cmake_executor_runner() {
  local backend_string_select="${1:-}"
  echo "Building executor_runner"
  rm -rf ${CMAKE_OUTPUT_DIR}
  mkdir ${CMAKE_OUTPUT_DIR}
  # Common options:
  COMMON="-DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE"
  if [[ "$backend_string_select" == "XNNPACK" ]]; then
    echo "Backend $backend_string_select selected"
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        ${COMMON} \
        -B${CMAKE_OUTPUT_DIR} .
    cmake --build ${CMAKE_OUTPUT_DIR} -j4
  elif [[ "$backend_string_select" == "CUDA" ]]; then
    echo "Backend $backend_string_select selected"
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_CUDA=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        ${COMMON} \
        -B${CMAKE_OUTPUT_DIR} .
    cmake --build ${CMAKE_OUTPUT_DIR} -j4
  else
    cmake -DCMAKE_BUILD_TYPE=Debug \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        ${COMMON} \
        -B${CMAKE_OUTPUT_DIR} .
    cmake --build ${CMAKE_OUTPUT_DIR} -j4 --config Debug
  fi
}

run_portable_executor_runner() {
  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/portable/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    build_cmake_executor_runner
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./${MODEL_NAME}.pte"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

test_model() {
  if [[ "${MODEL_NAME}" == "llama2" ]]; then
    # Install requirements for export_llama
    bash examples/models/llama/install_requirements.sh
    # Test export_llm script: python3 -m extension.llm.export.export_llm
    "${PYTHON_EXECUTABLE}" -m extension.llm.export.export_llm base.model_class="${MODEL_NAME}" base.checkpoint=examples/models/llama/params/demo_rand_params.pth base.params=examples/models/llama/params/demo_config.json
    run_portable_executor_runner
    rm "./${MODEL_NAME}.pte"
  fi
  STRICT="--strict"
  if [[ "${MODEL_NAME}" == "llava" ]]; then
    # Install requirements for llava
    bash examples/models/llava/install_requirements.sh
    STRICT="--no-strict"
  fi
  if [[ "${MODEL_NAME}" == "qwen2_5_1_5b" ]]; then
      # Install requirements for export_llama
      bash examples/models/llama/install_requirements.sh
      # Test export_llm script: python3 -m extension.llm.export.export_llm.
      # Use Llama random checkpoint with Qwen 2.5 1.5b model configuration.
      "${PYTHON_EXECUTABLE}" -m extension.llm.export.export_llm base.model_class="${MODEL_NAME}" base.params=examples/models/qwen2_5/config/1_5b_config.json
      rm "./${MODEL_NAME}.pte"
      return  # Skip running with portable executor runnner since portable doesn't support Qwen's biased linears.
  fi
  if [[ "${MODEL_NAME}" == "phi_4_mini" ]]; then
      # Install requirements for export_llama
      bash examples/models/llama/install_requirements.sh
      # Test export_llm script: python3 -m extension.llm.export.export_llm.
      "${PYTHON_EXECUTABLE}" -m extension.llm.export.export_llm base.model_class="${MODEL_NAME}" base.params=examples/models/phi_4_mini/config/config.json
      run_portable_executor_runner
      rm "./${MODEL_NAME}.pte"
      return
  fi

  # Export a basic .pte and run the model.
  "${PYTHON_EXECUTABLE}" -m examples.portable.scripts.export --model_name="${MODEL_NAME}" "${STRICT}"
  run_portable_executor_runner
}

test_model_with_xnnpack() {
  WITH_QUANTIZATION=$1
  WITH_DELEGATION=$2

  # Quantization-only
  if [[ ${WITH_QUANTIZATION} == true ]] && [[ ${WITH_DELEGATION} == false ]]; then
    bash examples/xnnpack/quantization/test_quantize.sh "${BUILD_TOOL}" "${MODEL_NAME}"
    return 0
  fi

  # Delegation and test with pybindings
  if [[ ${WITH_QUANTIZATION} == true ]]; then
    SUFFIX="q8"
    "${PYTHON_EXECUTABLE}" -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate --quantize --test_after_export
  else
    SUFFIX="fp32"
    "${PYTHON_EXECUTABLE}" -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate --test_after_export
  fi

  OUTPUT_MODEL_PATH="${MODEL_NAME}_xnnpack_${SUFFIX}.pte"
  EXPORTED_MODEL=${OUTPUT_MODEL_PATH}

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    # TODO eventually buck should also use consolidated executor runners
    buck2 run //examples/xnnpack:xnn_executor_runner -- --model_path "${OUTPUT_MODEL_PATH}"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    build_cmake_executor_runner "XNNPACK"
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "${OUTPUT_MODEL_PATH}"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

test_model_with_qnn() {
  source "$(dirname "${BASH_SOURCE[0]}")/build-qnn-sdk.sh"
  echo "ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
  echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
  echo "EXECUTORCH_ROOT: $EXECUTORCH_ROOT"

  export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/
  export PYTHONPATH=$EXECUTORCH_ROOT/..

  EXTRA_FLAGS=""
  # Ordered by the folder name, then alphabetically by the model name
  # Following models are inside examples/qualcomm/scripts folder
  if [[ "${MODEL_NAME}" == "dl3" ]]; then
    EXPORT_SCRIPT=deeplab_v3
  elif [[ "${MODEL_NAME}" == "edsr" ]]; then
    EXPORT_SCRIPT=edsr
    # Additional deps for edsr
    pip install piq
  elif [[ "${MODEL_NAME}" == "ic3" ]]; then
    EXPORT_SCRIPT=inception_v3
  elif [[ "${MODEL_NAME}" == "ic4" ]]; then
    EXPORT_SCRIPT=inception_v4
  elif [[ "${MODEL_NAME}" == "mb" ]]; then
    EXPORT_SCRIPT=mobilebert_fine_tune
    EXTRA_FLAGS="--num_epochs 1"
    pip install scikit-learn
  elif [[ "${MODEL_NAME}" == "mv2" ]]; then
    EXPORT_SCRIPT=mobilenet_v2
  elif [[ "${MODEL_NAME}" == "mv3" ]]; then
    EXPORT_SCRIPT=mobilenet_v3
  elif [[ "${MODEL_NAME}" == "vit" ]]; then
    EXPORT_SCRIPT=torchvision_vit
  elif [[ "${MODEL_NAME}" == "w2l" ]]; then
    EXPORT_SCRIPT=wav2letter
  elif [[ "${MODEL_NAME}" == "edsr" ]]; then
    EXPORT_SCRIPT=edsr
    # Additional deps for edsr
    pip install piq
  # Following models are inside examples/qualcomm/oss_scripts folder
  elif [[ "${MODEL_NAME}" == "albert" ]]; then
    EXPORT_SCRIPT=albert
  elif [[ "${MODEL_NAME}" == "bert" ]]; then
    EXPORT_SCRIPT=bert
  elif [[ "${MODEL_NAME}" == "conv_former" ]]; then
    EXPORT_SCRIPT=conv_former
  elif [[ "${MODEL_NAME}" == "cvt" ]]; then
    EXPORT_SCRIPT=cvt
  elif [[ "${MODEL_NAME}" == "distilbert" ]]; then
    EXPORT_SCRIPT=distilbert
  elif [[ "${MODEL_NAME}" == "dit" ]]; then
    EXPORT_SCRIPT=dit
  elif [[ "${MODEL_NAME}" == "efficientnet" ]]; then
    EXPORT_SCRIPT=efficientnet
  elif [[ "${MODEL_NAME}" == "eurobert" ]]; then
    EXPORT_SCRIPT=eurobert
  elif [[ "${MODEL_NAME}" == "focalnet" ]]; then
    EXPORT_SCRIPT=focalnet
  elif [[ "${MODEL_NAME}" == "mobilevit_v1" ]]; then
    EXPORT_SCRIPT=mobilevit_v1
  elif [[ "${MODEL_NAME}" == "mobilevit_v2" ]]; then
    EXPORT_SCRIPT=mobilevit_v2
  elif [[ "${MODEL_NAME}" == "pvt" ]]; then
    EXPORT_SCRIPT=pvt
  elif [[ "${MODEL_NAME}" == "roberta" ]]; then
    EXPORT_SCRIPT=roberta
  elif [[ "${MODEL_NAME}" == "swin" ]]; then
    EXPORT_SCRIPT=swin_transformer
  else
    echo "Unsupported model $MODEL_NAME"
    exit 1
  fi

  # Use SM8450 for S22, SM8550 for S23, and SM8560 for S24
  # TODO(guangyang): Make QNN chipset matches the target device
  QNN_CHIPSET=SM8450

  SCRIPT_FOLDER=""
  case "${MODEL_NAME}" in
    "dl3"|"mv3"|"mv2"|"ic4"|"ic3"|"vit"|"mb"|"w2l")
        SCRIPT_FOLDER=scripts
        ;;
    "cvt"|"dit"|"focalnet"|"mobilevit_v2"|"pvt"|"swin")
        SCRIPT_FOLDER=oss_scripts
        ;;
    "albert"|"bert"|"conv_former"|"distilbert"|"roberta"|"efficientnet"|"mobilevit_v1")
        pip install evaluate
        SCRIPT_FOLDER=oss_scripts
        # 16bit models will encounter op validation fail on some operations,
        # which requires CHIPSET >= SM8550.
        QNN_CHIPSET=SM8550
        ;;
    *)
        echo "Unsupported model $MODEL_NAME"
        exit 1
        ;;
  esac

  "${PYTHON_EXECUTABLE}" -m examples.qualcomm.${SCRIPT_FOLDER}.${EXPORT_SCRIPT} -b ${CMAKE_OUTPUT_DIR} -m ${QNN_CHIPSET} --ci --compile_only $EXTRA_FLAGS
  EXPORTED_MODEL=$(find "./${EXPORT_SCRIPT}" -type f -name "${MODEL_NAME}*.pte" -print -quit)
}

# Run CoreML tests.
#
# @param should_test If true, build and test the model using the coreml_executor_runner.
test_model_with_coreml() {
  local should_test="$1"
  local test_with_pybindings="$2"
  local dtype="$3"

  if [[ "${BUILD_TOOL}" != "cmake" ]]; then
    echo "coreml only supports cmake."
    exit 1
  fi

  RUN_WITH_PYBINDINGS=""
  if [[ "${test_with_pybindings}" == true ]]; then
    echo \"Running with pybindings\"
    export RUN_WITH_PYBINDINGS="--run_with_pybindings"
  fi

  "${PYTHON_EXECUTABLE}" -m examples.apple.coreml.scripts.export --model_name="${MODEL_NAME}" --compute_precision ${dtype} --use_partitioner ${RUN_WITH_PYBINDINGS}
  EXPORTED_MODEL=$(find "." -type f -name "${MODEL_NAME}*.pte" -print -quit)

  if [ -n "$EXPORTED_MODEL" ]; then
    echo "OK exported model: $EXPORTED_MODEL"
  else
    echo "[error] failed to export model: no .pte file found"
    exit 1
  fi

  # Run the model
  if [ "${should_test}" = true ]; then
    echo "Installing requirements needed to build coreml_executor_runner..."
    backends/apple/coreml/scripts/install_requirements.sh

    echo "Testing exported model with coreml_executor_runner..."
    local out_dir=$(mktemp -d)
    COREML_EXECUTOR_RUNNER_OUT_DIR="${out_dir}" examples/apple/coreml/scripts/build_executor_runner.sh
    "${out_dir}/coreml_executor_runner" --model_path "${EXPORTED_MODEL}"
  fi
}

test_model_with_mps() {
  "${PYTHON_EXECUTABLE}" -m examples.apple.mps.scripts.mps_example --model_name="${MODEL_NAME}" --use_fp16
  EXPORTED_MODEL=$(find "." -type f -name "${MODEL_NAME}*.pte" -print -quit)
}

test_model_with_mediatek() {
  if [[ "${MODEL_NAME}" == "dl3" ]]; then
    EXPORT_SCRIPT=deeplab_v3
  elif [[ "${MODEL_NAME}" == "mv3" ]]; then
    EXPORT_SCRIPT=mobilenet_v3
  elif [[ "${MODEL_NAME}" == "mv2" ]]; then
    EXPORT_SCRIPT=mobilenet_v2
  elif [[ "${MODEL_NAME}" == "ic4" ]]; then
    EXPORT_SCRIPT=inception_v4
  elif [[ "${MODEL_NAME}" == "ic3" ]]; then
    EXPORT_SCRIPT=inception_v3
  fi

  PYTHONPATH=examples/mediatek/ "${PYTHON_EXECUTABLE}" -m examples.mediatek.model_export_scripts.${EXPORT_SCRIPT} -d /tmp/neuropilot/train -a ${EXPORT_SCRIPT}
  EXPORTED_MODEL=$(find "./${EXPORT_SCRIPT}" -type f -name "*.pte" -print -quit)
}

test_model_with_cuda() {
  # Export a basic .pte and .ptd, then run the model.
  "${PYTHON_EXECUTABLE}" -m examples.cuda.scripts.export --model_name="${MODEL_NAME}" --output_dir "./"
  build_cmake_executor_runner "CUDA"
  ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./${MODEL_NAME}.pte" --data_path "./aoti_cuda_blob.ptd"
}


if [[ "${BACKEND}" == "portable" ]]; then
  echo "Testing ${MODEL_NAME} with portable kernels..."
  test_model
elif [[ "${BACKEND}" == *"qnn"* ]]; then
  echo "Testing ${MODEL_NAME} with qnn..."
  test_model_with_qnn
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
elif [[ "${BACKEND}" == *"coreml"* ]]; then
  echo "Testing ${MODEL_NAME} with coreml..."
  should_test_coreml=false
  if [[ "${BACKEND}" == *"test"* ]]; then
    should_test_coreml=true
  fi
  test_with_pybindings=false
  if [[ "${BACKEND}" == *"pybind"* ]]; then
    test_with_pybindings=true
  fi
  dtype=float16
  if [[ "${BACKEND}" == *"float32"* ]]; then
    dtype=float32
  fi
  test_model_with_coreml "${should_test_coreml}" "${test_with_pybindings}" "${dtype}"
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
elif [[ "${BACKEND}" == *"mps"* ]]; then
  echo "Testing ${MODEL_NAME} with mps..."
  test_model_with_mps
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
elif [[ "${BACKEND}" == *"xnnpack"* ]]; then
  echo "Testing ${MODEL_NAME} with xnnpack..."
  WITH_QUANTIZATION=true
  WITH_DELEGATION=true
  if [[ "$MODEL_NAME" == "mobilebert" ]]; then
    # TODO(T197452682)
    WITH_QUANTIZATION=false
  fi
  test_model_with_xnnpack "${WITH_QUANTIZATION}" "${WITH_DELEGATION}"
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
elif [[ "${BACKEND}" == "mediatek" ]]; then
  echo "Testing ${MODEL_NAME} with mediatek..."
  test_model_with_mediatek
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
elif [[ "${BACKEND}" == "cuda" ]]; then
  echo "Testing ${MODEL_NAME} with cuda..."
  test_model_with_cuda
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload
  fi
else
  set +e
  if [[ "${BACKEND}" == *"quantization"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization only..."
    test_model_with_xnnpack true false || Q_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK delegation only..."
    test_model_with_xnnpack false true || D_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"quantization"* ]] && [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization and delegation..."
    test_model_with_xnnpack true true || Q_D_ERROR="error"
    echo "::endgroup::"
  fi
  set -e
  if [[ -n "${Q_ERROR:-}" ]] || [[ -n "${D_ERROR:-}" ]] || [[ -n "${Q_D_ERROR:-}" ]]; then
    echo "Portable q8 ${Q_ERROR:-ok}," "Delegation fp32 ${D_ERROR:-ok}," "Delegation q8 ${Q_D_ERROR:-ok}"
    exit 1
  else
    prepare_artifacts_upload
  fi
fi
