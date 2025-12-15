#!/usr/bin/env bash

# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

help() {
    echo "Usage:"
    echo " $0 [TESTNAME]"
    echo "Without TESTNAME all tests will run; otherwise choose one of:"
    # This will list all lines in this file that is starting with test_ remove () { and print it as a list.
    # e,g, "test_pytest() { # Test ops and other things" -> test_pytest # Test ops and other things
    grep "^run_" $0 | sed 's/([^)]*)[[:space:]]*{*//g'
    exit
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    help
fi

if [[ $# -eq 0 ]]; then
    TEST_SUITES=(run_ootb_tests_ethos_u run_ootb_tests_tosa run_deit_e2e_ethos_u)
else
    TEST_SUITES=("$1")
fi


run_ootb_tests_ethos_u() {
    echo "$FUNCNAME: Running out-of-the-box tests for Arm Ethos-U"
    jupyter nbconvert \
        --to notebook \
        --execute examples/arm/ethos_u_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_ootb_tests_tosa() {
    echo "$FUNCNAME: Running out-of-the-box tests for TOSA"
    jupyter nbconvert \
        --to notebook \
        --execute backends/arm/scripts/TOSA_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_deit_e2e_ethos_u() {
    echo "$FUNCNAME: Fine-tune, export, build, and run the DEiT e2e test"

    local script_dir
    script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
    et_root_dir=$(cd "${script_dir}/../../.." && pwd)
    local example_dir="${et_root_dir}/examples/arm/image_classification_example"
    local work_root="${et_root_dir}/arm_test/deit_tiny_ootb_smoke"
    local model_dir="${work_root}/deit_tiny_finetuned"
    local export_dir="${work_root}/export"
    local build_dir="${work_root}/simple_app_deit_tiny"
    local image_path="${work_root}/dog.bmp"
    local pte_path="${export_dir}/deit_tiny_smoke.pte"
    local toolchain_file="${et_root_dir}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake"
    echo "${FUNCNAME}: Work root is ${work_root}; existing artifacts will be reused if present"

    mkdir -p "${model_dir}" "${export_dir}" "${build_dir}"

    setup_path_script=${et_root_dir}/examples/arm/arm-scratch/setup_path.sh
    source ${setup_path_script}

    source ${et_root_dir}/backends/arm/scripts/utils.sh
    local n_proc="$(get_parallel_jobs)"

    # Build ExecuTorch
    echo "${FUNCNAME}: Building ExecuTorch (if needed)"
    cmake --preset arm-baremetal -B "${et_root_dir}/cmake-out-arm"
    cmake --build "${et_root_dir}/cmake-out-arm" --target install -j"$n_proc"

    # Get and finetune model
    echo "${FUNCNAME}: Running DeiT fine-tuning script"
    python3 "${example_dir}/model_export/train_deit.py" \
        --output-dir "${model_dir}" \
        --num-epochs 1

    # Export model to pte
    local final_model_dir="${model_dir}/final_model"
    echo "${FUNCNAME}: Exporting quantized PTE from ${final_model_dir}"
    python3 "${example_dir}/model_export/export_deit.py" \
        --model-path "${final_model_dir}" \
        --output-path "${pte_path}" \
        --num-calibration-samples 100

    [[ -f "${pte_path}" ]] || {
        echo "${FUNCNAME}: Missing PTE at ${pte_path}"
        return 1
    }

    # Download demo image for inference
    local image_url="https://gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit/-/raw/main/resources/img_class/samples/dog.bmp?ref_type=heads"
    if [[ ! -f "${image_path}" ]]; then
        echo "${FUNCNAME}: Downloading sample image from ${image_url}"
        wget -O "${image_path}" "${image_url}"
    else
        echo "${FUNCNAME}: Reusing sample image at ${image_path}"
    fi

    # Build application
    echo "${FUNCNAME}: Configuring the minimal application"
    cmake \
        -S "${example_dir}/runtime" \
        -B "${build_dir}" \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
        -DET_PTE_FILE_PATH="${pte_path}" \
        -DIMAGE_PATH="${image_path}" \
        -DET_BUILD_DIR_PATH="${et_root_dir}/cmake-out-arm"

    echo "${FUNCNAME}: Building img_class_example"
    cmake --build "${build_dir}" -j"$n_proc" --target img_class_example

    # Run application on FVP
    local fvp_bin="${FVP_BINARY:-FVP_Corstone_SSE-320}"
    local elf="${build_dir}/img_class_example"

    echo "${FUNCNAME}: Running on ${fvp_bin}"
    "${fvp_bin}" \
        -C mps4_board.subsystem.ethosu.num_macs=256 \
        -C mps4_board.visualisation.disable-visualisation=1 \
        -C vis_hdlcd.disable_visualisation=1 \
        -C mps4_board.telnetterminal0.start_telnet=0 \
        -C mps4_board.uart0.out_file="-" \
        -C mps4_board.uart0.shutdown_on_eot=1 \
        -a "${elf}" \
        -C mps4_board.subsystem.ethosu.extra_args="--fast"

    echo "${FUNCNAME}: PASS"
}

for suite in "${TEST_SUITES[@]}"; do
    "${suite}"
done
