# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import pickle

from executorch.exir.serialize import (
    deserialize_from_flatbuffer,
    serialize_to_flatbuffer,
)
from executorch.exir.tests.common import get_test_program


def test() -> None:
    program = get_test_program()

    serialized_flatbuffer_og = serialize_to_flatbuffer(program)

    with open("boop.pkl", "wb") as f:
        pickle.dump(program, f)

    save_serialized_model("boop.pkl", "moop.ff")

    with open("moop.ff", "rb") as f:
        serialized_flatbuffer_loaded = f.read()

    assert serialized_flatbuffer_og == serialized_flatbuffer_loaded
    assert program == deserialize_from_flatbuffer(serialized_flatbuffer_loaded)


def save_serialized_model(input_file: str, output_file: str) -> None:
    with open(input_file, "rb") as f:
        program = pickle.load(f)

    flatbuffer = serialize_to_flatbuffer(program)

    with open(output_file, "wb") as f:
        f.write(flatbuffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--test", help="Test the script", action="store_true")
    parser.add_argument(
        "-i",
        "--input_file",
        help="Input pickle file containing the emitted Flatbuffer PyObject",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output pickle file to store the serialized Flatbuffer program",
    )

    args: argparse.Namespace = parser.parse_args()

    if args.test:
        test()

    else:
        save_serialized_model(args.input_file, args.output_file)
