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
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
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

test_pytest_ops() { # Test ops and other things
    echo "${TEST_SUITE_NAME}: Run pytest"

    # Make sure to not run this tests on FVP by removing the elf builds,
    # as they are detected by the unit tests and used if they exists
    rm -Rf arm_test/arm_semihosting_executor_runner_corstone-300
    rm -Rf arm_test/arm_semihosting_executor_runner_corstone-320

    # Prepare for pytest
    backends/arm/scripts/build_executorch.sh

    # Run arm baremetal pytest tests without FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=10 backends/arm/test/ --ignore=backends/arm/test/models
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models() { # Test ops and other things
    echo "${TEST_SUITE_NAME}: Run pytest"

    # Make sure to not run this tests on FVP by removing the elf builds,
    # as they are detected by the unit tests and used if they exists
    rm -Rf arm_test/arm_semihosting_executor_runner_corstone-300
    rm -Rf arm_test/arm_semihosting_executor_runner_corstone-320

    # Prepare for pytest
    backends/arm/scripts/build_executorch.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    # Run arm baremetal pytest tests without FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest() { # Test ops and other things
    echo "${TEST_SUITE_NAME}: Run pytest"
    test_pytest_ops
    test_pytest_models
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_ops_ethosu_fvp() { # Same as test_pytest but also sometime verify using Corstone FVP
    echo "${TEST_SUITE_NAME}: Run pytest with fvp"

    # Prepare Corstone-3x0 FVP for pytest
    backends/arm/scripts/build_executorch.sh
    # Build semihosting version of the runner used by pytest testing. This builds:
    # arm_test/arm_semihosting_executor_runner_corstone-300
    # arm_test/arm_semihosting_executor_runner_corstone-320
    backends/arm/test/setup_testing.sh

    # Run arm baremetal pytest tests with FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=10  backends/arm/test/ --ignore=backends/arm/test/models
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_ethosu_fvp() { # Same as test_pytest but also sometime verify using Corstone FVP
    echo "${TEST_SUITE_NAME}: Run pytest with fvp"

    # Prepare Corstone-3x0 FVP for pytest
    backends/arm/scripts/build_executorch.sh
    # Build semihosting version of the runner used by pytest testing. This builds:
    # arm_test/arm_semihosting_executor_runner_corstone-300
    # arm_test/arm_semihosting_executor_runner_corstone-320
    backends/arm/test/setup_testing.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    # Run arm baremetal pytest tests with FVP
    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_ethosu_fvp() { # Same as test_pytest but also sometime verify using Corstone FVP
    echo "${TEST_SUITE_NAME}: Run pytest with fvp"
    test_pytest_ops_ethosu_fvp
    test_pytest_models_ethosu_fvp
    echo "${TEST_SUITE_NAME}: PASS"
}


test_pytest_ops_vkml() { # Same as test_pytest but also sometime verify using VKML runtime
    echo "${TEST_SUITE_NAME}: Run pytest operator tests with VKML runtime"

    backends/arm/test/setup_testing_vkml.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=10  backends/arm/test/ \
            --ignore=backends/arm/test/models -k _vgf_
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_models_vkml() { # Same as test_pytest but also sometime verify VKML runtime
    echo "${TEST_SUITE_NAME}: Run pytest model tests with VKML runtime"

    backends/arm/scripts/build_executorch.sh
    backends/arm/test/setup_testing_vkml.sh

    # Install model dependencies for pytest
    source backends/arm/scripts/install_models_for_test.sh

    pytest  --verbose --color=yes --numprocesses=auto --durations=0 backends/arm/test/models -k _vgf_
    echo "${TEST_SUITE_NAME}: PASS"
}

test_pytest_vkml() { # Same as test_pytest but also sometime verify VKML runtime
    echo "${TEST_SUITE_NAME}: Run pytest with VKML"
    test_pytest_ops_vkml
    test_pytest_models_vkml
    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_vkml() { # End to End model tests using run.sh
    echo "${TEST_SUITE_NAME}: Test VKML delegate examples with run.sh"

    echo "${TEST_SUITE_NAME}: Test VKML"
    out_folder="arm_test/test_run"
    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=add --output=${out_folder}/runner --bundleio
    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=mul --output=${out_folder}/runner --bundleio

    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=qadd --output=${out_folder}/runner --bundleio
    examples/arm/run.sh --et_build_root=${out_folder} --target=vgf --model_name=qops --output=${out_folder}/runner --bundleio

    echo "${TEST_SUITE_NAME}: PASS"
}

test_run_ethosu_fvp() { # End to End model tests using run.sh
    echo "${TEST_SUITE_NAME}: Test ethos-u delegate examples with run.sh"

    # TOSA quantized
    echo "${TEST_SUITE_NAME}: Test target TOSA"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=TOSA-1.0+INT --model_name=add
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=TOSA-1.0+INT --model_name=mul

    # Ethos-U55
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

    # Ethos-U85
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
    echo "${TEST_SUITE_NAME}: Test target Cortex-M55 (on a Ethos-U)"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=add --bundleio --no_delegate --select_ops_list="aten::add.out"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qadd --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qops --bundleio
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u55-128 --model_name=qops --bundleio --no_delegate --select_ops_list="aten::sub.out,aten::add.out,aten::mul.out"
    examples/arm/run.sh --et_build_root=arm_test/test_run --target=ethos-u85-128 --model_name=qops --bundleio

    echo "${TEST_SUITE_NAME}: PASS"
}

test_models_vkml() { # End to End model tests using model_test.py
    echo "${TEST_SUITE_NAME}: Test VKML delegated models with test_model.py"

    # Build common libs once
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --build_libs

    # VKML
    echo "${TEST_SUITE_NAME}: Test target VKML"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=vgf --model=resnet18 --extra_runtime_flags="--bundleio_atol=0.2 --bundleio_rtol=0.2"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=vgf --model=resnet50 --extra_runtime_flags="--bundleio_atol=0.2 --bundleio_rtol=0.2"

    echo "${TEST_SUITE_NAME}: PASS"
}

test_models_tosa() { # End to End model tests using model_test.py
    echo "${TEST_SUITE_NAME}: Test TOSA delegated models with test_model.py"

    # Build common libs once
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --build_libs

    # TOSA quantized
    echo "${TEST_SUITE_NAME}: Test ethos-u target TOSA"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=mv2
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=mv3
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=lstm
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=edsr
    # python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=emformer_transcribe # Takes long time to run
    # python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=emformer_join       # Takes long time to run
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=w2l
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=ic3
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=ic4
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=resnet18
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=TOSA-1.0+INT --model=resnet50

    echo "${TEST_SUITE_NAME}: PASS"
}

test_models_ethos-u55() { # End to End model tests using model_test.py
    echo "${TEST_SUITE_NAME}: Test Ethos-U55 delegated models with test_model.py"

    # Build common libs once
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --build_libs

    # Ethos-U55
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U55"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u55-128 --model=mv2  --extra_flags="-DET_ATOL=2.00 -DET_RTOL=2.00"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u55-64  --model=mv3  --extra_flags="-DET_ATOL=5.00 -DET_RTOL=5.00"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u55-256 --model=lstm --extra_flags="-DET_ATOL=0.03 -DET_RTOL=0.03"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u55-128 --model=resnet18 --extra_flags="-DET_ATOL=0.2 -DET_RTOL=0.2"
    # TODO: Output performance for resnet50 is bad with per-channel quantization (MLETORCH-1149).
    # Also we get OOM when running this model. Disable it for now.
    #python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u55-128 --model=resnet50 --extra_flags="-DET_ATOL=6.2 -DET_RTOL=6.2"

    echo "${TEST_SUITE_NAME}: PASS"
}

test_models_ethos-u85() { # End to End model tests using model_test.py
    echo "${TEST_SUITE_NAME}: Test Ethos-U85 delegated models with test_model.py"

    # Build common libs once
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --build_libs

    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U85"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-256 --model=mv2 --extra_flags="-DET_ATOL=2.00 -DET_RTOL=2.00"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-512 --model=mv3 --extra_flags="-DET_ATOL=5.00 -DET_RTOL=5.00"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-128 --model=lstm --extra_flags="-DET_ATOL=0.03 -DET_RTOL=0.03"
    #python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-128 --model=w2l --extra_flags="-DET_ATOL=0.01 -DET_RTOL=0.01"  # Takes long time to run
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-256 --model=ic4 --extra_flags="-DET_ATOL=0.8 -DET_RTOL=0.8" --timeout=2400
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-128 --model=resnet18 --extra_flags="-DET_ATOL=0.2 -DET_RTOL=0.2"
    python3 backends/arm/test/test_model.py --test_output=arm_test/test_model --target=ethos-u85-128 --model=resnet50 --extra_flags="-DET_ATOL=0.2 -DET_RTOL=0.2"

    echo "${TEST_SUITE_NAME}: PASS"
}


test_full_ethosu_fvp() { # All End to End model tests
    echo "${TEST_SUITE_NAME}: Test ethos-u delegate models and examples on fvp"

    test_run_ethosu_fvp
    test_models_tosa
    test_models_ethos-u55
    test_models_ethos-u85
    echo "${TEST_SUITE_NAME}: PASS"
}

test_full_vkml() { # All End to End model tests
    echo "${TEST_SUITE_NAME}: Test VGF delegate models and examples with VKML"

    test_run_vkml
    test_models_vkml
    echo "${TEST_SUITE_NAME}: PASS"
}

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
