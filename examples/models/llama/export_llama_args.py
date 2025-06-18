# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run export_llama with the legacy argparse setup.
"""

from .export_llama_lib import build_args_parser, export_llama


def main(args) -> None:
    parser = build_args_parser()
    args = parser.parse_args(args)
    export_llama(args)


if __name__ == "__main__":
    main()
