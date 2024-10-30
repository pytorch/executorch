# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Helper to determine whether a CI should run on a list of modified files.
Takes a list of modified files and a list of matchers.
If any modified file matches to the matcher, print '1' to indicate CI should
run.
"""


import re


def filename_matches(filename: str, matchers: list[str]):
    combined = "(" + ")|(".join(matchers) + ")"
    return re.match(combined, filename)


def any_match(modified_files: list[str], matchers: list[str]):
    return any(filename_matches(f, matchers) for f in modified_files)


def main(modified_file_list_path: str, matchers_path: str):
    with open(modified_file_list_path, "r") as f:
        modified_files = f.read().splitlines()
    with open(matchers_path, "r") as f:
        matchers = f.read().splitlines()
    if any_match(modified_files, matchers):
        print("1")
    else:
        print("0")


if __name__ == "__main__":
    main()
