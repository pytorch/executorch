#!/bin/bash
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Executorch root
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
cd "${et_root_dir}"
pwd


TEST_SUITE=$1

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
}

test_pytest() { # Test ops and other things
    echo "${TEST_SUITE_NAME}: Run pytest"
    cd "${et_root_dir}"
    source examples/arm/ethos-u-scratch/setup_path.sh
    backends/arm/scripts/build_quantized_ops_aot_lib.sh

    # Run arm baremetal pytest tests without FVP
    pytest  --verbose --color=yes --numprocesses=auto backends/arm/test/
}

test_pytest_ethosu_fvp() { # Same as test_pytest but also sometime verify using Corstone FVP
    echo "${TEST_SUITE_NAME}: Run pytest with fvp"

    source examples/arm/ethos-u-scratch/setup_path.sh

    # Prepare Corstone-3x0 FVP for pytest
    examples/arm/run.sh --model_name=add --build_only
    backends/arm/test/setup_testing.sh

    # Run arm baremetal pytest tests with FVP
    pytest  --verbose --color=yes --numprocesses=auto backends/arm/test/ --arm_run_corstoneFVP
}

test_run_ethosu_fvp() { # End to End model tests
    echo "${TEST_SUITE_NAME}: Test ethos-u delegate examples with run.sh"

    source examples/arm/ethos-u-scratch/setup_path.sh

    # TOSA quantized
    echo "${TEST_SUITE_NAME}: Test ethos-u target TOSA"
    examples/arm/run.sh --target=TOSA --model_name=mv2
    examples/arm/run.sh --target=TOSA --model_name=lstm
    examples/arm/run.sh --target=TOSA --model_name=edsr

    # Ethos-U55
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U55"
    examples/arm/run.sh --target=ethos-u55-128 --model_name=mv2
    examples/arm/run.sh --target=ethos-u55-128 --model_name=lstm

    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U85"
    examples/arm/run.sh --target=ethos-u85-128 --model_name=mv2
    examples/arm/run.sh --target=ethos-u85-128 --model_name=lstm
    }

${TEST_SUITE}