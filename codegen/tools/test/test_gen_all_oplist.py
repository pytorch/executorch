# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import executorch.codegen.tools.gen_all_oplist as gen_all_oplist
import yaml


class TestGenAllOplist(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test1_yaml = os.path.join(self.temp_dir.name, "test1.yaml")
        with open(self.test1_yaml, "w") as f:
            f.write(
                """
build_features: []
custom_classes: []
et_kernel_metadata:
  aten::_cdist_forward.out:
  - default
  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out:
  - default
operators:
  aten::_cdist_forward.out:
    debug_info:
    - test1.yaml
    include_all_overloads: false
    is_root_operator: true
    is_used_for_training: true
  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out:
    debug_info:
    - test1.yaml
    include_all_overloads: false
    is_root_operator: true
    is_used_for_training: true
            """
            )
        self.test2_yaml = os.path.join(self.temp_dir.name, "test2.yaml")
        with open(self.test2_yaml, "w") as f:
            f.write(
                """
build_features: []
custom_classes: []
et_kernel_metadata:
  aten::_cdist_forward.out:
  - default
  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out:
  - default
operators:
  aten::_cdist_forward.out:
    debug_info:
    - test2.yaml
    include_all_overloads: false
    is_root_operator: true
    is_used_for_training: true
  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out:
    debug_info:
    - test2.yaml
    include_all_overloads: false
    is_root_operator: true
    is_used_for_training: true
            """
            )

    def test_gen_all_oplist_with_1_valid_yaml(self) -> None:
        args = [
            f"--output_dir={self.temp_dir.name}",
            f"--model_file_list_path={self.test1_yaml}",
        ]
        gen_all_oplist.main(args)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "selected_operators.yaml"))
        )
        with open(os.path.join(self.temp_dir.name, "selected_operators.yaml")) as f:
            es = yaml.safe_load(f)
        debug_info = es["operators"]["aten::_cdist_forward.out"]["debug_info"]
        self.assertEqual(len(debug_info), 1)
        self.assertTrue("test1.yaml" in debug_info)

    def test_gen_all_oplist_with_2_conflicting_yaml_no_check(
        self,
    ) -> None:
        file_ = tempfile.NamedTemporaryFile()
        with open(file_.name, "w") as f:
            f.write(f"{self.test1_yaml}\n{self.test2_yaml}")
        args = [
            f"--output_dir={self.temp_dir.name}",
            f"--model_file_list_path=@{file_.name}",
        ]
        gen_all_oplist.main(args)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "selected_operators.yaml"))
        )
        with open(os.path.join(self.temp_dir.name, "selected_operators.yaml")) as f:
            es = yaml.safe_load(f)
        debug_info = es["operators"]["aten::_cdist_forward.out"]["debug_info"]
        self.assertEqual(len(debug_info), 2)
        self.assertTrue(self.test1_yaml in debug_info)
        self.assertTrue(self.test2_yaml in debug_info)

    def test_gen_all_oplist_with_2_conflicting_yaml_raises(
        self,
    ) -> None:
        file_ = tempfile.NamedTemporaryFile()
        with open(file_.name, "w") as f:
            f.write(f"{self.test1_yaml}\n{self.test2_yaml}")
        args = [
            f"--output_dir={self.temp_dir.name}",
            f"--model_file_list_path=@{file_.name}",
            "--check_ops_not_overlapping",
        ]
        with self.assertRaisesRegex(Exception, "Operator .* is used in 2 models"):
            gen_all_oplist.main(args)

    def test_gen_all_oplist_merges_files_and_directories(self) -> None:
        model_dir = os.path.join(self.temp_dir.name, "model2")
        os.mkdir(model_dir)
        os.rename(self.test2_yaml, os.path.join(model_dir, "selected_operators.yaml"))
        # A selection that selects nothing, such as one with only prim ops.
        empty_yaml = os.path.join(self.temp_dir.name, "empty.yaml")
        with open(empty_yaml, "w") as f:
            f.write("build_features: []\n")
        out_dir = os.path.join(self.temp_dir.name, "out")
        os.mkdir(out_dir)
        gen_all_oplist.gen_all_oplist(out_dir, [self.test1_yaml, model_dir, empty_yaml])
        with open(os.path.join(out_dir, "selected_operators.yaml")) as f:
            es = yaml.safe_load(f)
        debug_info = es["operators"]["aten::_cdist_forward.out"]["debug_info"]
        self.assertEqual(
            sorted(debug_info),
            sorted(
                [
                    os.path.realpath(self.test1_yaml),
                    os.path.realpath(
                        os.path.join(model_dir, "selected_operators.yaml")
                    ),
                ]
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()
