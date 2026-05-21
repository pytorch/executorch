#!/usr/bin/env bash

# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd "${script_dir}/../../.." && pwd)
ootb_requirements="${et_root_dir}/backends/arm/requirements-arm-ootb-test.txt"

cd "${et_root_dir}"

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
    TEST_SUITES=(run_ootb_tests_ethos_u run_ootb_tests_tosa run_ootb_tests_vgf run_deit_e2e_ethos_u run_swin2sr_e2e_vgf)
else
    TEST_SUITES=("$1")
fi


install_ootb_test_requirements() {
    python3 - <<'PY' || pip install -r "${ootb_requirements}"
import notebook  # noqa: F401
import nbconvert  # noqa: F401
PY
}


run_ootb_tests_ethos_u() {
    echo "$FUNCNAME: Running out-of-the-box tests for Arm Ethos-U"
    install_ootb_test_requirements
    jupyter nbconvert \
        --to notebook \
        --execute examples/arm/ethos_u_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_ootb_tests_tosa() {
    echo "$FUNCNAME: Running out-of-the-box tests for TOSA"
    install_ootb_test_requirements
    jupyter nbconvert \
        --to notebook \
        --execute backends/arm/scripts/TOSA_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_ootb_tests_vgf() {
    echo "$FUNCNAME: Running out-of-the-box tests for VGF"
    install_ootb_test_requirements
    jupyter nbconvert \
        --to notebook \
        --execute examples/arm/vgf_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_deit_e2e_ethos_u() {
    echo "$FUNCNAME: Fine-tune, export, build, and run the DEiT e2e test"

    local example_dir="${et_root_dir}/examples/arm/image_classification_example_ethos_u"
    local work_root="${et_root_dir}/arm_test/deit_tiny_ootb_smoke"
    local model_dir="${work_root}/deit_tiny_finetuned"
    local export_dir="${work_root}/export"
    local build_dir="${work_root}/simple_app_deit_tiny"
    local image_path="${work_root}/dog.bmp"
    local pte_path="${export_dir}/deit_tiny_smoke.pte"
    local toolchain_file="${et_root_dir}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake"
    echo "${FUNCNAME}: Work directory: ${work_root}; existing artifacts will be reused if present"

    mkdir -p "${model_dir}" "${export_dir}" "${build_dir}"

    setup_path_script=${et_root_dir}/examples/arm/arm-scratch/setup_path.sh
    source ${setup_path_script}

    source ${et_root_dir}/backends/arm/scripts/utils.sh
    local n_proc="$(get_parallel_jobs)"

    # Build ExecuTorch
    echo "${FUNCNAME}: Building ExecuTorch (if needed)"
    cmake --preset arm-baremetal -B "${et_root_dir}/cmake-out-arm"
    cmake --build "${et_root_dir}/cmake-out-arm" --target install -j"$n_proc"

    # Install requirements
    pip install -r examples/arm/image_classification_example_ethos_u/requirements.txt

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
        if command -v wget >/dev/null 2>&1; then
            wget -O "${image_path}" "${image_url}"
        elif command -v curl >/dev/null 2>&1; then
            curl -L "${image_url}" -o "${image_path}"
        else
            echo "${FUNCNAME}: Missing wget or curl; unable to download sample image"
            return 1
        fi
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

run_swin2sr_e2e_vgf() {
    echo "$FUNCNAME: Prepare demo assets, export FP/INT8, build, and run the Swin2SR VGF e2e test"

    local script_dir
    script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
    et_root_dir=$(cd "${script_dir}/../../.." && pwd)
    local example_dir="${et_root_dir}/examples/arm/super_resolution_example_vgf"
    local work_root="${et_root_dir}/arm_test/swin2sr_vgf_ootb_smoke"
    local demo_dir="${work_root}/demo_assets"
    local runtime_dir="${demo_dir}/runtime"
    local runner_path="${work_root}/executor_runner"
    local input_image="${runtime_dir}/demo_lr_64.png"
    local fp_pte_path="${demo_dir}/swin2sr_x2_vgf_fp.pte"
    local int8_pte_path="${demo_dir}/swin2sr_x2_vgf_int8.pte"
    local fp_output_image="${runtime_dir}/demo_fp_128.png"
    local int8_output_image="${runtime_dir}/demo_int8_128.png"
    local checkpoint_id="caidas/swin2SR-classical-sr-x2-64"
    local checkpoint_revision="cee1c923c6a37361c6e5650b65dcf4be821e5d52"
    echo "${FUNCNAME}: Work directory: ${work_root}; existing artifacts will be reused if present"

    mkdir -p "${demo_dir}" "${runtime_dir}"

    setup_path_script=${et_root_dir}/examples/arm/arm-scratch/setup_path.sh
    source ${setup_path_script}

    echo "${FUNCNAME}: Installing example requirements"
    pip install -r "${example_dir}/requirements.txt"

    echo "${FUNCNAME}: Preparing deterministic demo assets"
    python3 "${example_dir}/model_export/prepare_demo_assets.py" \
        --output-dir "${demo_dir}"

    echo "${FUNCNAME}: Building VKML executor_runner"
    "${et_root_dir}/backends/arm/scripts/build_executor_runner_vkml.sh" \
        --output="${work_root}"

    if [[ ! -f "${runner_path}" ]]; then
        runner_path=$(find "${work_root}" -name executor_runner -type f | head -n 1)
    fi
    [[ -f "${runner_path}" ]] || {
        echo "${FUNCNAME}: Missing executor_runner under ${work_root}"
        return 1
    }

    echo "${FUNCNAME}: Exporting FP Swin2SR model"
    python3 "${example_dir}/model_export/export_super_resolution.py" \
        --model-name swin2sr \
        --checkpoint "${checkpoint_id}" \
        --checkpoint-revision "${checkpoint_revision}" \
        --input-height 64 \
        --input-width 64 \
        --quantization-mode none \
        --eval-lr-dir "${demo_dir}/eval/lr" \
        --eval-hr-dir "${demo_dir}/eval/hr" \
        --num-eval-samples 2 \
        --output-path "${fp_pte_path}"

    for artifact in \
        "${fp_pte_path}" \
        "${demo_dir}/swin2sr_x2_vgf_fp.json" \
        "${demo_dir}/swin2sr_x2_vgf_fp_delegation.txt" \
        "${demo_dir}/swin2sr_x2_vgf_fp_metrics.json"; do
        [[ -f "${artifact}" ]] || {
            echo "${FUNCNAME}: Missing FP export artifact ${artifact}"
            return 1
        }
    done

    echo "${FUNCNAME}: Exporting INT8 Swin2SR model"
    python3 "${example_dir}/model_export/export_super_resolution.py" \
        --model-name swin2sr \
        --checkpoint "${checkpoint_id}" \
        --checkpoint-revision "${checkpoint_revision}" \
        --input-height 64 \
        --input-width 64 \
        --quantization-mode int8 \
        --calibration-lr-dir "${demo_dir}/calibration/lr" \
        --eval-lr-dir "${demo_dir}/eval/lr" \
        --eval-hr-dir "${demo_dir}/eval/hr" \
        --num-calibration-samples 4 \
        --num-eval-samples 2 \
        --output-path "${int8_pte_path}"

    for artifact in \
        "${int8_pte_path}" \
        "${demo_dir}/swin2sr_x2_vgf_int8.json" \
        "${demo_dir}/swin2sr_x2_vgf_int8_delegation.txt" \
        "${demo_dir}/swin2sr_x2_vgf_int8_metrics.json"; do
        [[ -f "${artifact}" ]] || {
            echo "${FUNCNAME}: Missing INT8 export artifact ${artifact}"
            return 1
        }
    done

    echo "${FUNCNAME}: Running FP runtime smoke"
    python3 "${example_dir}/runtime/run_super_resolution.py" \
        --model-path "${fp_pte_path}" \
        --runner "${runner_path}" \
        --input-image "${input_image}" \
        --output-image "${fp_output_image}" \
        --working-dir "${runtime_dir}/fp_work"

    [[ -f "${fp_output_image}" ]] || {
        echo "${FUNCNAME}: Missing FP runtime output ${fp_output_image}"
        return 1
    }

    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "${FUNCNAME}: Running INT8 runtime smoke"
        python3 "${example_dir}/runtime/run_super_resolution.py" \
            --model-path "${int8_pte_path}" \
            --runner "${runner_path}" \
            --input-image "${input_image}" \
            --output-image "${int8_output_image}" \
            --working-dir "${runtime_dir}/int8_work"

        [[ -f "${int8_output_image}" ]] || {
            echo "${FUNCNAME}: Missing INT8 runtime output ${int8_output_image}"
            return 1
        }
    else
        # TODO: MLETORCH-2105 remove this once the next ML SDK release supports
        # quantized VKML runtime validation on Darwin.
        echo "${FUNCNAME}: Skipping INT8 runtime on $(uname -s); quantized VKML runtime validation is Linux-only"
    fi

    echo "${FUNCNAME}: PASS"
}

for suite in "${TEST_SUITES[@]}"; do
    "${suite}"
done
