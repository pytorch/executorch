# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write <op>_<scheme>.ptn for each op and quantization scheme that fits a case
of the op."""

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from executorch.backends.native.test.samples.ops import ops
from executorch.backends.native.test.samples.record import record
from executorch.backends.native.test.samples.schemes import schemes


def _record(name: str, scheme: str, out: Path) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    op = ops()[name]
    record(op.cases(), schemes()[scheme], out / f"{name}_{scheme}.ptn", op.dtypes)


def main() -> None:
    all_ops, all_schemes = ops(), schemes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ops", nargs="+", choices=all_ops, default=list(all_ops))
    parser.add_argument(
        "--schemes", nargs="+", choices=all_schemes, default=list(all_schemes)
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Forked workers can't use the CUDA state torch may have initialized here.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=spawn) as pool:
        jobs = [
            pool.submit(_record, name, scheme, args.out)
            for name in args.ops
            for scheme in args.schemes
        ]
    for job in jobs:
        job.result()


if __name__ == "__main__":
    main()
