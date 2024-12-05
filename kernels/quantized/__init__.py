# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

try:
    import glob

    import torch as _torch
    import executorch

    # Ideally package is installed in only one location but usage of
    # PYATHONPATH can result in multiple locations.
    # ATM this is mainly used in CI for qnn runner. Will need to revisit this
    executorch_package_path = executorch.__path__[-1]
    libs = list(
        glob.glob(
            f"{executorch_package_path}/**/libquantized_ops_aot_lib.*", recursive=True
        )
    )
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    logging.info(f"Loading custom ops library: {libs[0]}")
    _torch.ops.load_library(libs[0])
    del _torch
except:
    import logging

    logging.info("libquantized_ops_aot_lib is not loaded")
    del logging
