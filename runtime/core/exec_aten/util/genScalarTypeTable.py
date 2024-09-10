# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

indexToType = [
    "U1",
    "I1",
    "I2",
    "I4",
    "I8",
    "F2",
    "F4",
    "F8",
    "C2",
    "C4",
    "C8",
    "B1",
    "BF",
]
promoteTypesLookup = [
    ["U1", "I2", "I2", "I4", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "U1", "BF"],
    ["I2", "I1", "I2", "I4", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "I1", "BF"],
    ["I2", "I2", "I2", "I4", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "I2", "BF"],
    ["I4", "I4", "I4", "I4", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "I4", "BF"],
    ["I8", "I8", "I8", "I8", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "I8", "BF"],
    ["F2", "F2", "F2", "F2", "F2", "F2", "F4", "F8", "C2", "C4", "C8", "F2", "F4"],
    ["F4", "F4", "F4", "F4", "F4", "F4", "F4", "F8", "C4", "C4", "C8", "F4", "F4"],
    ["F8", "F8", "F8", "F8", "F8", "F8", "F8", "F8", "C8", "C8", "C8", "F8", "F8"],
    ["C2", "C2", "C2", "C2", "C2", "C2", "C4", "C8", "C2", "C4", "C8", "C2", "C4"],
    ["C4", "C4", "C4", "C4", "C4", "C4", "C4", "C8", "C4", "C4", "C8", "C4", "C4"],
    ["C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8", "C8"],
    ["U1", "I1", "I2", "I4", "I8", "F2", "F4", "F8", "C2", "C4", "C8", "B1", "BF"],
    ["BF", "BF", "BF", "BF", "BF", "F4", "F4", "F8", "C4", "C4", "C8", "BF", "BF"],
]
for rowIndex, row in enumerate(promoteTypesLookup):
    for colIndex, col in enumerate(row):
        print(f"TABLE_ENTRY({indexToType[rowIndex]}, {indexToType[colIndex]}, {col});")
