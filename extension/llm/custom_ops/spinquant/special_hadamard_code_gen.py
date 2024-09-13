# Portions (c) Meta Platforms, Inc. and affiliates.
# This file is adapted from
# https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/code_gen.py .

# BSD 3-Clause License

# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

import numpy as np

# From https://en.wikipedia.org/wiki/Paley_construction (construction II for q = 5)

had_12_paley = """
+-++++++++++
--+-+-+-+-+-
+++-++----++
+---+--+-++-
+++++-++----
+-+---+--+-+
++--+++-++--
+--++---+--+
++----+++-++
+--+-++---+-
++++----+++-
+-+--+-++---
"""

# From http://neilsloane.com/hadamard/

had_12 = """
+-----------
++-+---+++-+
+++-+---+++-
+-++-+---+++
++-++-+---++
+++-++-+---+
++++-++-+---
+-+++-++-+--
+--+++-++-+-
+---+++-++-+
++---+++-++-
+-+---+++-++
"""

had_20_will = """
+----+----++--++-++-
-+----+---+++---+-++
--+----+---+++-+-+-+
---+----+---+++++-+-
----+----++--++-++-+
-+++++-----+--+++--+
+-+++-+---+-+--+++--
++-++--+---+-+--+++-
+++-+---+---+-+--+++
++++-----++--+-+--++
--++-+-++-+-----++++
---++-+-++-+---+-+++
+---++-+-+--+--++-++
++---++-+----+-+++-+
-++---++-+----+++++-
-+--+--++-+----+----
+-+-----++-+----+---
-+-+-+---+--+----+--
--+-+++------+----+-
+--+--++------+----+
"""


had_28_will = """
+------++----++-+--+-+--++--
-+-----+++-----+-+--+-+--++-
--+-----+++---+-+-+----+--++
---+-----+++---+-+-+-+--+--+
----+-----+++---+-+-+++--+--
-----+-----++++--+-+--++--+-
------++----++-+--+-+--++--+
--++++-+-------++--+++-+--+-
---++++-+-----+-++--+-+-+--+
+---+++--+----++-++--+-+-+--
++---++---+----++-++--+-+-+-
+++---+----+----++-++--+-+-+
++++--------+-+--++-++--+-+-
-++++--------+++--++--+--+-+
-+-++-++--++--+--------++++-
+-+-++--+--++--+--------++++
-+-+-++--+--++--+----+---+++
+-+-+-++--+--+---+---++---++
++-+-+-++--+------+--+++---+
-++-+-+-++--+------+-++++---
+-++-+---++--+------+-++++--
-++--++-+-++-+++----++------
+-++--++-+-++-+++-----+-----
++-++---+-+-++-+++-----+----
-++-++-+-+-+-+--+++-----+---
--++-++++-+-+----+++-----+--
+--++-+-++-+-+----+++-----+-
++--++-+-++-+-+----++------+
"""


had_40_tpal = """
+-------------------+-------------------
++-++----+-+-++++--+++-++----+-+-++++--+
+++-++----+-+-++++--+++-++----+-+-++++--
+-++-++----+-+-++++-+-++-++----+-+-++++-
+--++-++----+-+-+++++--++-++----+-+-++++
++--++-++----+-+-+++++--++-++----+-+-+++
+++--++-++----+-+-+++++--++-++----+-+-++
++++--++-++----+-+-+++++--++-++----+-+-+
+++++--++-++----+-+-+++++--++-++----+-+-
+-++++--++-++----+-++-++++--++-++----+-+
++-++++--++-++----+-++-++++--++-++----+-
+-+-++++--++-++----++-+-++++--++-++----+
++-+-++++--++-++----++-+-++++--++-++----
+-+-+-++++--++-++---+-+-+-++++--++-++---
+--+-+-++++--++-++--+--+-+-++++--++-++--
+---+-+-++++--++-++-+---+-+-++++--++-++-
+----+-+-++++--++-+++----+-+-++++--++-++
++----+-+-++++--++-+++----+-+-++++--++-+
+++----+-+-++++--++-+++----+-+-++++--++-
+-++----+-+-++++--+++-++----+-+-++++--++
+--------------------+++++++++++++++++++
++-++----+-+-++++--+--+--++++-+-+----++-
+++-++----+-+-++++-----+--++++-+-+----++
+-++-++----+-+-++++--+--+--++++-+-+----+
+--++-++----+-+-++++-++--+--++++-+-+----
++--++-++----+-+-+++--++--+--++++-+-+---
+++--++-++----+-+-++---++--+--++++-+-+--
++++--++-++----+-+-+----++--+--++++-+-+-
+++++--++-++----+-+------++--+--++++-+-+
+-++++--++-++----+-+-+----++--+--++++-+-
++-++++--++-++----+---+----++--+--++++-+
+-+-++++--++-++----+-+-+----++--+--++++-
++-+-++++--++-++------+-+----++--+--++++
+-+-+-++++--++-++----+-+-+----++--+--+++
+--+-+-++++--++-++---++-+-+----++--+--++
+---+-+-++++--++-++--+++-+-+----++--+--+
+----+-+-++++--++-++-++++-+-+----++--+--
++----+-+-++++--++-+--++++-+-+----++--+-
+++----+-+-++++--++----++++-+-+----++--+
+-++----+-+-++++--++-+--++++-+-+----++--
"""

# NOTE: the original Dao-AILab/fast-hadamard-transform uses had_12_paley rather than
# had_12 here. However, SpinQuant and QuaRot seem to use had_12, so we follow them here.
had_strings = [had_12, had_20_will, had_28_will, had_40_tpal]

header = """

#pragma once

"""


TEMPLATE = """
__device__ __forceinline__ void hadamard_mult_thread_{N}(float x[{N}]) {{
    float out[{N}];
    {code}
    #pragma unroll
    for (int i = 0; i < {N}; i++) {{ x[i] = out[i]; }}
}}

"""


CPU_TEMPLATE = """
template <typename T>
void hadamard_mult_{N}(T* x) {{
    float out[{N}];
    {code}
    #pragma unroll
    for (int i = 0; i < {N}; i++) {{ x[i] = out[i]; }}
}}

"""

STRIDED_CPU_TEMPLATE = """
template <typename T>
void hadamard_mult_{N}_strided(T* input, int stride) {{
    T x[{N}];
    T out[{N}];
    {strided_load_code}
    {code}
    #pragma unroll
    for (int ii = 0; ii < {N}; ++ii) {{ input[stride * ii] = out[ii]; }}
}}

"""


def string_to_array(string):
    # Convert strings of + and - to bool arrays
    string = string.strip().replace("+", "1").replace("-", "-1").split()
    return np.stack(
        [
            np.fromstring(" ".join(string[i]), dtype=np.int32, sep=" ")
            for i in range(len(string))
        ]
    )


def strided_load_code_gen(N):
    return "\n    ".join([f"x[{i}] = input[{i} * stride];" for i in range(N)])


def array_code_gen(arr, template):
    N = arr.shape[0]
    assert arr.shape[0] == arr.shape[1]
    out = []
    for i in range(N):
        out.append(
            f"out[{i}] = "
            + " ".join([f"{'+' if arr[i, j] == 1 else '-'} x[{j}]" for j in range(N)])
            + ";"
        )
    return template.format(
        N=str(N), code="\n    ".join(out), strided_load_code=strided_load_code_gen(N)
    )


OPTION_TO_TEMPLATE = {
    "cuda": TEMPLATE,
    "cpu": CPU_TEMPLATE,
    "strided_cpu": STRIDED_CPU_TEMPLATE,
}


def main(option="cuda"):
    try:
        template = OPTION_TO_TEMPLATE[option]
    except KeyError:
        raise Exception(
            f"bad target option {option}; options are {', '.join(OPTION_TO_TEMPLATE.keys())}"
        )
    output_dir = Path(__file__).parent / "fast_hadamard_transform_special.h"
    generated_line = f"// @{'generated'} by special_hadamard_code_gen.py {option}\n"

    output_dir.write_text(
        generated_line
        + header
        + "".join(array_code_gen(string_to_array(s), template) for s in had_strings)
    )


if __name__ == "__main__":
    import sys

    option = "cuda"
    if len(sys.argv) > 1:
        option = sys.argv[1]
    main(option)
