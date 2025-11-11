# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.tflite_generator.meta.meta import CustomOptions


class FlexTranspose(CustomOptions):

    def __init__(self) -> None:
        super().__init__(
            "FlexTranspose",
            bytearray(
                [
                    9,
                    84,
                    114,
                    97,
                    110,
                    115,
                    112,
                    111,
                    115,
                    101,
                    0,
                    39,
                    18,
                    9,
                    84,
                    114,
                    97,
                    110,
                    115,
                    112,
                    111,
                    115,
                    101,
                    26,
                    0,
                    26,
                    0,
                    42,
                    11,
                    10,
                    5,
                    84,
                    112,
                    101,
                    114,
                    109,
                    18,
                    2,
                    48,
                    3,
                    42,
                    7,
                    10,
                    1,
                    84,
                    18,
                    2,
                    48,
                    1,
                    50,
                    0,
                    0,
                    2,
                    52,
                    42,
                    20,
                    20,
                    4,
                    40,
                    1,
                ]
            ),
        )
