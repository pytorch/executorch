# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from export_preprocess_lib import export_preprocess, lower_to_executorch_preprocess


def main():
    ep = export_preprocess()
    et = lower_to_executorch_preprocess(ep)

    with open("preprocess.pte", "wb") as file:
        et.write_to_file(file)


if __name__ == "__main__":
    main()
