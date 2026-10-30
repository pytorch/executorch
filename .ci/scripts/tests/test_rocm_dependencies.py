# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class RocmDependenciesTest(unittest.TestCase):
    def test_compiled_torchao_uses_the_selected_torch_and_propagates_failures(self):
        stubs = """
conda() { echo "RUNTIME $*"; }
python() {
  if [[ "$1" == "-c" ]]; then
    if [[ "$2" == *torch_pin* ]]; then echo 2.15.0
    elif [[ "$2" == *rocm_sdk* ]]; then echo 7.14.1
    else echo /env; fi
  elif [[ "$*" == "-m rocm_sdk path --root" ]]; then
    echo /sdk
  elif [[ "$1" == "-" ]]; then
    cat >/dev/null
    echo VERIFY
    return "$VERIFY_STATUS"
  else
    echo "PIP $* CPP=${USE_CPP:-unset} CPU=${USE_CPU_KERNELS:-unset} JOBS=${MAX_JOBS:-unset}"
    if [[ "$*" == *third-party/ao* ]]; then return "$AO_STATUS"; fi
    if [[ "$*" == *torch==* ]]; then return "$TORCH_STATUS"; fi
    if [[ "$*" == *'rocm[devel]'* ]]; then return "$SDK_STATUS"; fi
  fi
}
"""
        for torch_status, sdk_status, ao_status, verify_status in (
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ):
            with self.subTest(
                torch=torch_status, sdk=sdk_status, ao=ao_status, verify=verify_status
            ):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        stubs
                        + "source .ci/scripts/install-rocm-dependencies.sh\necho INSTALL_ET\necho SDK_HOME=$ROCM_HOME HIP_PATH=$HIP_PATH ROCM_PATH=$ROCM_PATH\necho CC=$CC CXX=$CXX",
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "ROCM_VERSION": "7.14",
                        "PYTORCH_ROCM_INDEX": "https://download.pytorch.org/whl/rocm7.14",
                        "TORCH_STATUS": str(torch_status),
                        "SDK_STATUS": str(sdk_status),
                        "AO_STATUS": str(ao_status),
                        "VERIFY_STATUS": str(verify_status),
                        "MAX_JOBS": "8",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    max(torch_status, sdk_status, ao_status, verify_status),
                    result.stderr,
                )
                self.assertTrue(
                    result.stdout.startswith(
                        "RUNTIME install -y --override-channels -c conda-forge "
                        "gxx_linux-64=16.2.0 libstdcxx-ng=16.2.0 libgcc-ng=16.2.0 "
                        "sysroot_linux-64=2.28\n"
                    )
                )
                self.assertIn(
                    "torch==2.15.0+rocm7.14 --index-url https://download.pytorch.org/whl/rocm7.14",
                    result.stdout,
                )
                if not torch_status:
                    self.assertIn(
                        "rocm[devel]==7.14.1 --index-url https://download.pytorch.org/whl/rocm7.14",
                        result.stdout,
                    )
                if not (torch_status or sdk_status):
                    self.assertIn(
                        "--no-build-isolation --no-deps third-party/ao CPP=1 CPU=1 JOBS=8",
                        result.stdout,
                    )
                else:
                    self.assertNotIn("third-party/ao", result.stdout)
                self.assertEqual(
                    "VERIFY" in result.stdout,
                    not (torch_status or sdk_status or ao_status),
                )
                self.assertEqual("INSTALL_ET" in result.stdout, result.returncode == 0)
                if result.returncode == 0:
                    self.assertIn(
                        "SDK_HOME=/sdk HIP_PATH=/sdk ROCM_PATH=/sdk", result.stdout
                    )
                    self.assertIn(
                        "CC=/env/bin/x86_64-conda-linux-gnu-cc "
                        "CXX=/env/bin/x86_64-conda-linux-gnu-c++",
                        result.stdout,
                    )


if __name__ == "__main__":
    unittest.main()
