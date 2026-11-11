# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.vulkan.test.op_tests.cases import (
    register_test_suite,
    test_suites,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_benchmark_vk import (
    VkBenchmarkFileGen,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_computegraph import (
    ComputeGraphGen,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_correctness_base import (
    init_list_str,
)
from executorch.backends.vulkan.test.op_tests.utils.gen_correctness_vk import (
    VkCorrectnessTestGen,
)
from executorch.backends.vulkan.test.op_tests.utils.test_suite import VkTestSuite
from torchgen.model import BackendIndex, DispatchKey, Location, NativeFunction


def native_function(schema):
    function, _ = NativeFunction.from_yaml(
        {"func": schema, "variants": "function"},
        Location(__file__, 1),
        valid_tags=set(),
    )
    return function


class GeneratorTest(unittest.TestCase):
    def test_suite_exposes_only_the_padding_extension(self):
        suite = VkTestSuite([])
        self.assertFalse(suite.check_zero_padding)
        for field in [
            "input_sequences",
            "symint_args",
            "arg_transposes",
            "expected_build_error",
        ]:
            self.assertNotIn(field, vars(suite))

    def test_registration_accumulates_without_an_option(self):
        name = "test.registry.default"
        other = "test.other_registry.default"
        first, second, third = VkTestSuite([]), VkTestSuite([]), VkTestSuite([])

        def first_suite():
            return first

        def more_suites():
            return [second, third]

        def skipped_suite():
            self.fail("Skipped registration must not invoke its getter")

        try:
            register_test_suite(name)(first_suite)
            register_test_suite([name, other])(more_suites)
            self.assertEqual(len(test_suites[name]), 3)
            self.assertIs(test_suites[name][0], first)
            self.assertIs(test_suites[name][1], second)
            self.assertIs(test_suites[name][2], third)
            self.assertEqual(len(test_suites[other]), 2)
            self.assertIs(
                register_test_suite(name, skip=True)(skipped_suite), skipped_suite
            )
            self.assertEqual(len(test_suites[name]), 3)
        finally:
            test_suites.pop(name, None)
            test_suites.pop(other, None)

    def test_string_defaults_do_not_include_schema_quotes(self):
        function = native_function(
            "gelu(Tensor self, *, str approximate='none') -> Tensor"
        )
        suite = VkTestSuite([((3,),)])
        code = VkCorrectnessTestGen(
            "aten.gelu.default", function, suite
        ).generate_suite_cpp()
        self.assertIn('std::string_view("none")', code)
        self.assertNotIn("std::string_view(\"'none'\")", code)

    def test_scalar_shape_is_an_empty_initializer(self):
        self.assertEqual(init_list_str(()), "{}")
        self.assertEqual(init_list_str([]), "{}")
        self.assertEqual(init_list_str([()]), "{{}}")

    def test_optional_string_arguments(self):
        function = native_function(
            "div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode='floor') -> Tensor"
        )
        for inputs, expected in [
            (((3, 7), (3, 7)), 'std::string_view("floor")'),
            (((3, 7), (3, 7), None), "std::nullopt"),
        ]:
            with self.subTest(inputs=inputs):
                suite = VkTestSuite([inputs])
                try:
                    code = VkCorrectnessTestGen(
                        "aten.div.Tensor_mode", function, suite
                    ).generate_suite_cpp()
                except RuntimeError as error:
                    self.fail(f"Optional string generation failed: {error}")
                self.assertIn(expected, code)
                self.assertIn(".has_value()", code)
                self.assertIn("add_string(std::string(", code)

    def setUp(self):
        self.previous_backend = ComputeGraphGen.backend_key
        ComputeGraphGen.backend_key = BackendIndex(
            dispatch_key=DispatchKey.CPU,
            use_out_as_primary=True,
            device_guard=False,
            external=False,
            index={},
        )

    def tearDown(self):
        ComputeGraphGen.backend_key = self.previous_backend

    def test_padding_is_opt_in(self):
        function = native_function("neg(Tensor self) -> Tensor")
        suite = VkTestSuite([((2, 3, 5),)])
        generator = ComputeGraphGen("aten.neg.default", function, suite)
        self.assertNotIn("OutputPaddingCheck", generator.gen_op_check_fn())
        suite.check_zero_padding = True
        code = generator.gen_op_check_fn()
        self.assertIn("OutputPaddingCheck out_ref_padding", code)
        self.assertIn("out_ref_padding.poison", code)
        self.assertIn("out_ref_padding.check", code)
        self.assertLess(code.index(".poison"), code.index("->execute()"))
        self.assertLess(code.index("->execute()"), code.index(".check"))

    def test_benchmark_execution_has_no_padding_checks(self):
        function = native_function("neg(Tensor self) -> Tensor")
        suite = VkTestSuite([((2, 3, 5),)])
        suite.check_zero_padding = True
        generator = ComputeGraphGen("aten.neg.default", function, suite)
        self.assertNotIn("OutputPaddingCheck", generator.gen_build_graph_fn())
        self.assertNotIn("_padding", generator.gen_op_exec_graph_fn())
        file_generator = VkBenchmarkFileGen("unused")
        file_generator.add_suite("aten.neg.default", function, suite)
        self.assertEqual(len(file_generator.suites_gens), 1)
