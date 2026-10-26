# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.samsung.serialization.compile_options import (
    ENN_COMPILE_OPTION_TITLE,
    gen_samsung_backend_compile_spec,
    SamsungChipset,
)


class CompileOptionsTest(unittest.TestCase):
    def test_supported_chipsets_serialize_to_nonempty_compile_specs(self) -> None:
        supported = {
            SamsungChipset.E9945,
            SamsungChipset.E9955,
            SamsungChipset.E9965,
        }
        self.assertEqual(
            supported,
            set(SamsungChipset) - {SamsungChipset.UNDEFINED_CHIP_V},
        )

        for chipset in supported:
            with self.subTest(chipset=chipset.name):
                compile_spec = gen_samsung_backend_compile_spec(chipset.name)
                self.assertEqual(ENN_COMPILE_OPTION_TITLE, compile_spec.key)
                self.assertTrue(compile_spec.value)

    def test_chipset_name_is_case_insensitive(self) -> None:
        self.assertEqual(
            gen_samsung_backend_compile_spec("E9945"),
            gen_samsung_backend_compile_spec("e9945"),
        )


if __name__ == "__main__":
    unittest.main()
