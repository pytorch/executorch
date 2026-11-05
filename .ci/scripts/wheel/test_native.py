# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install a repaired Native wheel into a clean environment and test GGML."""

import argparse
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    source = args.source.resolve()
    fixtures = args.fixtures.resolve()
    with tempfile.TemporaryDirectory(prefix="native-wheel-test-") as directory:
        work = Path(directory)
        venv.create(work / "venv", with_pip=True)
        python = work / "venv/bin/python"
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            env.pop(name, None)
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel)],
            cwd=work,
            env=env,
            check=True,
            timeout=600,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            cwd=work,
            env=env,
            check=True,
            timeout=60,
        )
        shutil.copytree(fixtures, work / "fixtures")
        shutil.copyfile(
            source / "backends/native/pybindings/test/test_bindings.py",
            work / "test_bindings.py",
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
import importlib.metadata
import importlib.util
from pathlib import Path
import numpy as np
from executorch_native import Model

for name in ('torch', 'executorch'):
    assert importlib.util.find_spec(name) is None, name
installed = {d.metadata['Name'].lower() for d in importlib.metadata.distributions()}
assert installed <= {'pip', 'setuptools', 'executorch-native', 'numpy'}, installed
for name in ('add', 'conv_bn'):
    package = Path('fixtures') / f'{name}.ptn'
    with np.load(package.with_suffix('.npz')) as data:
        session = Model.load(package).create_session('ggml')
        (actual,) = session.run([data['input']])
        np.testing.assert_allclose(actual, data['expected'], rtol=1e-4, atol=1e-4)
    print(f'PASS: {package.name} with GGML')
""",
            ],
            cwd=work,
            env=env,
            check=True,
            timeout=120,
        )
        env["NATIVE_TEST_PACKAGE"] = str(work / "fixtures/add.ptn")
        env["NATIVE_TEST_ENGINE"] = "ggml"
        subprocess.run(
            [str(python), "-m", "unittest", "-v", "test_bindings"],
            cwd=work,
            env=env,
            check=True,
            timeout=300,
        )
    print(f"PASS: clean install of {wheel.name}")


if __name__ == "__main__":
    main()
