# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .export_llama import build_model


def main() -> None:
    output = build_model(
        modelname="language_llama", extra_opts="--fairseq2 -Q", par_local_output=False
    )
    print(f"Built executorch language llama in {output}")


if __name__ == "__main__":
    main()  # pragma: no cover
