#!/bin/bash
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Rename this script

set -e

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Executorch root
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
cd "${et_root_dir}"
pwd
scratch_dir=${et_root_dir}/examples/arm/arm-scratch
setup_path_script=${scratch_dir}/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."


TEST_SUITE=$1

# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }
source ${setup_path_script}

help() {
    echo "Usage:"
    echo " $0 <TESTNAME>"
    echo " where <TESTNAME> can be any of:"
    # This will list all lines in this file that is starting with test_ remove () { and print it as a list.
    # e,g, "test_pytest() { # Test ops and other things" -> test_pytest # Test ops and other things
    echo "all # run all tests"
    grep "^test_" $0 | sed 's/([^)]*)[[:space:]]*{*//g'
    exit
}

if [[ -z "${TEST_SUITE:-}" ]]; then
    echo "Missing test suite name, exiting..."
    help
else
    echo "Run Arm baremetal test suite ${TEST_SUITE}"
fi

TEST_SUITE_NAME="$(basename "$0") ${TEST_SUITE}"

all() { # Run all tests
    # This will list all lines in this file that is starting with test_ remove () { and add this script name in
    # front of it and execute it in a sub shell
    # e.g. from this file:
    #
    # test_pytest() { # Test ops and other things
    #  bla bla bla
    # }
    # test_pytest_ethosu_fvp() { # Same as test_pytest but ...
    #  bla bla bla
    # }
    #...
    # become a small script:
    # ----
    # backends/arm/test/test_arm_baremetal.sh test_pytest # Test ops and other things
    # backends/arm/test/test_arm_baremetal.sh test_pytest_ethosu_fvp # Same as test_pytest but ...
    # ...
    # ----
    # That is executed
    echo "${TEST_SUITE_NAME}: Run all tests"
    grep "^test_" backends/arm/test/test_arm_baremetal.sh | sed 's/([^)]*)[[:space:]]*{*//g' | sed "s|^|$0 |" | sh
    echo "${TEST_SUITE_NAME}: PASS"
}

# -------------------------------------------
# -------- Non target-specific tests --------
# -------------------------------------------
test_pytest_ops_no_target() {
    echo "${TEST_SUITE_NAME}: Run pytest ops for target-less tests"

    # Run arm baremetal pytest tests without target
    pytest  --verbose --color=yes --numprocesses=auto --durations=10 backends/arm/test/ --ignore=backends/arm/test/models -k no_target
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_no_target() {
    echo "${TEST_SUITE_NAME}: Run pytest models for target-less tests"

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    # Run arm baremetal pytest tests without FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k no_target
    echo "${TEST_SUITE_NAME}: PASS"
}

# -------------------------------------
# -------- TOSA specific tests --------
# -------------------------------------
test_pytest_ops_tosa() {
    echo "${TEST_SUITE_NAME}: Run pytest ops for TOSA"

    pytest  --verbose --color=yes --numprocesses=auto --durations=10 backends/arm/test/ --ignore=backends/arm/test/models -k tosa
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_tosa() {
    echo "${TEST_SUITE_NAME}: Run pytest models for TOSA"

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k tosa
    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_tosa() {
    echo "${TEST_SUITE_NAME}: Test TOSA delegate examples with run.sh"

    echo "${TEST_SUITE_NAME}: Test target TOSA"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=TOSA-1.0+INT --model_name=add
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=TOSA-1.0+INT --model_name=mul

    echo "${TEST_SUITE_NAME}: PASS"
}

# ----------------------------------------------
# -------- Arm Ethos-U55 specific tests --------
# ----------------------------------------------
test_pytest_ops_ethos_u55() {
    echo "${TEST_SUITE_NAME}: Run pytest ops for Arm Ethos-U55"

    backends/arm/scripts/build_executorch.sh
    backends/arm/test/setup_testing.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=10  backends/arm/test/ --ignore=backends/arm/test/models -k u55
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_ethos_u55() {
    echo "${TEST_SUITE_NAME}: Run pytest models for Arm Ethos-U55"

    backends/arm/scripts/build_executorch.sh
    backends/arm/test/setup_testing.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k u55
    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_ethos_u55() {
    echo "${TEST_SUITE_NAME}: Test ethos-u55 delegate examples with run.sh"

    echo "${TEST_SUITE_NAME}: Test target Ethos-U55"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-64 --model_name=add
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-256 --model_name=add --bundleio --etdump
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --etdump
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=mul
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --pte_placement=elf
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-256 --model_name=add --pte_placement=0x38000000
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=mul --bundleio --pte_placement=elf
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=mul --bundleio --pte_placement=0x38000000
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --bundleio --pte_placement=0x38000000
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=examples/arm/example_modules/add.py
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=examples/arm/example_modules/add.py --bundleio

    # Cortex-M op tests
    echo "${TEST_SUITE_NAME}: Test target Cortex-M55 (on Ethos-U55)"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --bundleio --no_delegate --select_ops_list="aten::add.out"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qadd --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qops --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qops --bundleio --no_delegate --select_ops_list="aten::sub.out,aten::add.out,aten::mul.out"

    echo "${TEST_SUITE_NAME}: PASS"
}

# ----------------------------------------------
# -------- Arm Ethos-U85 specific tests --------
# ----------------------------------------------
test_pytest_ops_ethos_u85() {
    echo "${TEST_SUITE_NAME}: Run pytest ops for Arm Ethos-U85"

    backends/arm/scripts/build_executorch.sh
    backends/arm/test/setup_testing.sh

    # Run arm baremetal pytest tests with FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=10  backends/arm/test/ --ignore=backends/arm/test/models -k u85
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_ethos_u85() {
    echo "${TEST_SUITE_NAME}: Run pytest models for Arm Ethos-U85"

    backends/arm/scripts/build_executorch.sh
    backends/arm/test/setup_testing.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k u85
    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_ethos_u85() {
    echo "${TEST_SUITE_NAME}: Test ethos-u85 delegate examples with run.sh"

    echo "${TEST_SUITE_NAME}: Test target Ethos-U85"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=add
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-256 --model_name=add --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-512 --model_name=add --bundleio --etdump
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-1024 --model_name=add --etdump
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-2048 --model_name=mul --pte_placement=elf
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=mul --pte_placement=0x38000000
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=mul --bundleio --pte_placement=elf
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-256 --model_name=mul --bundleio --pte_placement=0x38000000
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=examples/arm/example_modules/add.py
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-1024 --model_name=examples/arm/example_modules/add.py --bundleio

    # Cortex-M op tests
    echo "${TEST_SUITE_NAME}: Test target Cortex-M55 (on Ethos-U85)"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=qops --bundleio

    echo "${TEST_SUITE_NAME}: PASS"
}

# ----------------------------------------------------------
# -------- Vulkan Graph Format (VGF) specific tests --------
# ----------------------------------------------------------
test_pytest_ops_vkml() {
    echo "${TEST_SUITE_NAME}: Run pytest operator tests with VKML runtime"

    source backends/arm/test/setup_testing_vkml.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=10  backends/arm/test/ \
            --ignore=backends/arm/test/models -k _vgf_
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_vkml() {
    echo "${TEST_SUITE_NAME}: Run pytest model tests with VKML runtime"

    source backends/arm/test/setup_testing_vkml.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k _vgf_
    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_vkml() {
    echo "${TEST_SUITE_NAME}: Test VKML delegate examples with run.sh"

    echo "${TEST_SUITE_NAME}: Test VKML"
    out_folder="arm_test/test_run"

    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=add --output=${out_folder}/runner
    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=mul --output=${out_folder}/runner

    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=qadd --output=${out_folder}/runner
    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=qops --output=${out_folder}/runner

    echo "${TEST_SUITE_NAME}: PASS"
}

# ------------------------------------
# -------- Miscelaneous tests --------
# ------------------------------------
test_model_smollm2-135M() {
    echo "${TEST_SUITE_NAME}: Test SmolLM2-135M on Ethos-U85"

    # Build common libs once
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --build_libs

    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-128 --model=smollm2 --extra_flags="-DEXECUTORCH_SELECT_OPS_LIST=dim_order_ops::_to_dim_order_copy.out"

    echo "${TEST_SUITE_NAME}: PASS"


}

test_smaller_stories_llama() {
    echo "${TEST_SUITE_NAME}: Test smaller_stories_llama"

    backends/arm/scripts/build_executorch.sh

    mkdir -p stories110M
    pushd stories110M
    wget -N https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
    echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
    popd

    # Get path to source directory
    pytest \
    -c /dev/null \
    --verbose \
    --color=yes \
    --numprocesses=auto \
    --log-level=DEBUG \
    --junit-xml=stories110M/test-reports/unittest.xml \
    -s \
    backends/arm/test/models/test_llama.py \
    --llama_inputs stories110M/stories110M.pt stories110M/params.json stories110m

    echo "${TEST_SUITE_NAME}: PASS"
}

test_memory_allocation() {
    echo "${TEST_SUITE_NAME}: Test ethos-u memory allocation with run.sh"

    mkdir -p arm_test/test_run
    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test target Ethos-U85"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=examples/arm/example_modules/add.py &> arm_test/test_run/full.log
    python3 backends/arm/test/test_memory_allocator_log.py --log arm_test/test_run/full.log \
            --require "model_pte_program_size" "<= 3000 B" \
            --require "method_allocator_planned" "<= 64 B" \
            --require "method_allocator_loaded" "<= 1024 B" \
            --require "method_allocator_input" "<= 16 B" \
            --require "Total DRAM used" "<= 0.06 KiB"
    echo "${TEST_SUITE_NAME}: PASS"
}

test_undefinedbehavior_sanitizer() {
    echo "${TEST_SUITE_NAME}: Test ethos-u executor_runner with UBSAN"

    mkdir -p arm_test/test_run
    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test target Ethos-U85"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=examples/arm/example_modules/add.py --build_type=UndefinedSanitizer
    echo "${TEST_SUITE_NAME}: PASS"
}

test_address_sanitizer() {
    echo "${TEST_SUITE_NAME}: Test ethos-u executor_runner with ASAN"

    mkdir -p arm_test/test_run
    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test target Ethos-U85"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=examples/arm/example_modules/add.py --build_type=AddressSanitizer
    echo "${TEST_SUITE_NAME}: PASS"
}


${TEST_SUITE}
