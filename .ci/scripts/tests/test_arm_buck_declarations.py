# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


def load_checker():
    path = Path(__file__).resolve().parents[1] / "check_arm_buck_declarations.py"
    spec = importlib.util.spec_from_file_location("check_arm_buck_declarations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_checker()
Declarations = checker.Declarations
Snapshot = checker.Snapshot
analyze = checker.analyze
compare = checker.compare
glob_match = checker.glob_match

SCOPE = "backends/arm/test"
TESTER = SCOPE + "/tester/runner.py"
BUCK = SCOPE + "/BUCK"
TARGETS = SCOPE + "/targets.bzl"


def library(name, sources, deps="[]"):
    return f"fbcode_target(_kind=runtime.python_library, name={name!r}, srcs={sources}, deps={deps})\n"


def fixture():
    return {
        BUCK: library(
            "arm_tester_lib",
            'glob(["tester/*.py"], exclude=["tester/excluded.py"])',
            '["//executorch/exir:umbrella"]',
        ),
        TESTER: "from executorch.exir.schema import Tensor\n",
        TARGETS: 'def define_arm_tests():\n    test_files = []\n    test_files += native.glob(["passes/test_*.py"])\n',
        "exir/BUCK": library("umbrella", "[]", '[":schema"]')
        + library("schema", '["schema.py"]'),
        "exir/schema.py": "class Tensor: pass\n",
        "devtools/pte_tool/BUCK": library(
            "diff_pte_lib", '["diff_pte.py"]', '["//executorch/exir:schema"]'
        )
        + 'runtime.python_binary(name="diff_pte", srcs=["diff_pte.py"])\n',
        "devtools/pte_tool/diff_pte.py": "class PTEDiffResult: pass\n",
    }


def regressions(before, after):
    return compare(Snapshot(before), Snapshot(after))[0]


# Each test covers an independent declaration regression.
# pylint: disable-next=too-many-public-methods
class DeclarationTests(unittest.TestCase):
    def test_known_missing_import_and_registration_then_metadata_fix(self):
        before = fixture()
        after = dict(before)
        after[
            TESTER
        ] += "from executorch.devtools.pte_tool.diff_pte import PTEDiffResult\n"
        after[SCOPE + "/tester/pte_diff_utils.py"] = (
            "from executorch.devtools.pte_tool.diff_pte import PTEDiffResult\n"
            "from executorch.exir.schema import Tensor\n"
        )
        test = SCOPE + "/misc/test_diff_pte.py"
        after[test] = "def test_diff_pte():\n    assert True\n"
        issues = regressions(before, after)
        self.assertEqual(3, len(issues))
        self.assertEqual(2, sum(i.kind == "missing dependency" for i in issues))
        self.assertTrue(
            any(i.kind == "unregistered test" and i.source == test for i in issues)
        )
        self.assertTrue(
            all(
                "diff_pte_lib" in i.detail
                for i in issues
                if i.kind == "missing dependency"
            )
        )
        after[BUCK] = library(
            "arm_tester_lib",
            'glob(["tester/*.py"], exclude=["tester/excluded.py"])',
            '["//executorch/exir:umbrella", "//executorch/devtools/pte_tool:diff_pte_lib"]',
        )
        after[TARGETS] += '    test_files += ["misc/test_diff_pte.py"]\n'
        self.assertEqual(set(), regressions(before, after))

    def test_transitive_dependency_and_cycle_are_valid(self):
        files = fixture()
        files["exir/BUCK"] += library("cycle", "[]", '[":umbrella"]')
        files["exir/BUCK"] = files["exir/BUCK"].replace(
            'deps=[":schema"]', 'deps=[":schema", ":cycle"]'
        )
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_dependency_deletion_without_python_change_fails(self):
        before = fixture()
        after = dict(before)
        after[BUCK] = library("arm_tester_lib", 'glob(["tester/*.py"])')
        issues = regressions(before, after)
        self.assertEqual(1, len(issues))
        self.assertIn("//executorch/exir:schema", next(iter(issues)).detail)

    def test_existing_gaps_do_not_mask_different_new_gap(self):
        before = fixture()
        before[BUCK] = library("arm_tester_lib", 'glob(["tester/*.py"])')
        after = dict(before)
        after[
            TESTER
        ] += "from executorch.devtools.pte_tool.diff_pte import PTEDiffResult\n"
        issues = regressions(before, after)
        self.assertEqual(1, len(issues))
        self.assertIn("diff_pte_lib", next(iter(issues)).detail)

    def test_python_binary_is_not_an_import_provider(self):
        files = fixture()
        files[TESTER] = (
            "from executorch.devtools.pte_tool.diff_pte import PTEDiffResult\n"
        )
        files["devtools/pte_tool/BUCK"] = (
            'runtime.python_binary(name="diff_pte", srcs=["diff_pte.py"])\n'
        )
        issues = analyze(Snapshot(files))
        self.assertTrue(
            any(
                i.kind == "cannot validate"
                and "no supported python_library provider" in i.detail
                for i in issues
            )
        )

    def test_internal_branch_is_not_union_of_incompatible_deps(self):
        files = fixture()
        files[BUCK] = library(
            "arm_tester_lib",
            'glob(["tester/*.py"])',
            '["//executorch/exir:umbrella"] if runtime.is_oss else []',
        )
        issues = analyze(Snapshot(files))
        self.assertTrue(any(i.kind == "missing dependency" for i in issues))
        files[BUCK] = files[BUCK].replace(
            "if runtime.is_oss else", "if not runtime.is_oss else"
        )
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_unknown_deps_fail_instead_of_becoming_empty(self):
        before = fixture()
        after = dict(before)
        after[BUCK] = library(
            "arm_tester_lib", 'glob(["tester/*.py"])', "choose_dependencies()"
        )
        issues = regressions(before, after)
        self.assertTrue(
            any(
                i.kind == "cannot validate" and "choose_dependencies" in i.detail
                for i in issues
            )
        )

    def test_unknown_sources_fail_instead_of_disappearing(self):
        before = fixture()
        after = dict(before)
        after[BUCK] = library("arm_tester_lib", "choose_sources()")
        issues = regressions(before, after)
        self.assertTrue(
            any(
                i.kind == "cannot validate" and "choose_sources" in i.detail
                for i in issues
            )
        )

    def test_unknown_conditional_rule_fails(self):
        before = fixture()
        after = dict(before)
        after[BUCK] += (
            'if native.read_config("mode", "value"):\n'
            '    runtime.python_library(name="conditional", srcs=["extra.py"])\n'
        )
        issues = regressions(before, after)
        self.assertTrue(
            any(
                i.kind == "cannot validate" and "conditional declaration" in i.detail
                for i in issues
            )
        )

    def test_helpers_are_not_registered_by_filename_alone(self):
        before = fixture()
        after = dict(before)
        after[SCOPE + "/misc/test_helpers.py"] = "def make_payload():\n    return b''\n"
        self.assertEqual(set(), regressions(before, after))
        after[
            SCOPE + "/misc/test_helpers.py"
        ] += "class TestPayload:\n    def test_payload(self):\n        pass\n"
        self.assertTrue(
            any(i.kind == "unregistered test" for i in regressions(before, after))
        )

    def test_unittest_class_and_glob_registered_pytest_function(self):
        files = fixture()
        files[SCOPE + "/passes/test_registered.py"] = (
            "def test_operation():\n    pass\n"
        )
        files[SCOPE + "/misc/unusual_name.py"] = (
            "import unittest\nclass Case(unittest.TestCase):\n    def test_stuff(self):\n        pass\n"
        )
        issues = analyze(Snapshot(files))
        self.assertEqual(1, len(issues))
        self.assertEqual(SCOPE + "/misc/unusual_name.py", next(iter(issues)).source)

    def test_type_checking_only_import_is_not_runtime_requirement(self):
        files = fixture()
        files[
            TESTER
        ] += "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from executorch.missing import Whatever\n"
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_glob_exclude_and_relative_intralibrary_import(self):
        files = fixture()
        files[TESTER] += "from .helper import answer\n"
        files[SCOPE + "/tester/helper.py"] = "answer = 42\n"
        files[SCOPE + "/tester/excluded.py"] = "from executorch.missing import Nope\n"
        self.assertEqual(set(), analyze(Snapshot(files)))
        self.assertFalse(glob_match("tester/sub/deep.py", "tester/*.py"))
        self.assertTrue(glob_match("tester/helper.py", "tester/**/*.py"))
        self.assertTrue(glob_match("tester/sub/deep.py", "tester/**/*.py"))

    def test_from_package_submodule_import(self):
        files = fixture()
        files[TESTER] = "from executorch.exir import schema\n"
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_only_called_local_declaration_helpers_create_providers(self):
        files = fixture()
        files["exir/BUCK"] = (
            'load(":targets.bzl", "define_common_targets")\ndefine_common_targets()\n'
        )
        files["exir/targets.bzl"] = (
            "def define_common_targets():\n    "
            + library("umbrella", "[]", '[":schema"]')
            + "    "
            + library("schema", '["schema.py"]')
            + "def define_unused():\n    "
            + library("ghost", '["ghost.py"]')
        )
        files["exir/ghost.py"] = "x = 1\n"
        declarations = Declarations(Snapshot(files))
        self.assertIsNone(declarations.get("//executorch/exir:ghost"))
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_boolean_expressions_short_circuit_and_preserve_operand_values(self):
        declarations = Declarations(Snapshot({}))
        for expression, expected in (
            ("runtime.is_oss and unsupported()", False),
            ("not runtime.is_oss or unsupported()", True),
            ('[] or ["schema.py"]', ["schema.py"]),
            ('["schema.py"] and []', []),
            ('[] or False or ["schema.py"]', ["schema.py"]),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(
                    expected,
                    declarations.value(
                        checker.ast.parse(expression, mode="eval").body, {}, SCOPE
                    ),
                )
        for expression in ("False or unsupported()", "True and unsupported()"):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(checker.Unsupported, "unsupported"):
                    declarations.value(
                        checker.ast.parse(expression, mode="eval").body, {}, SCOPE
                    )

    def test_short_circuit_condition_does_not_reject_valid_declarations(self):
        files = fixture()
        files[BUCK] = (
            'if not runtime.is_oss or native.read_config("mode", "value"):\n'
            + "    "
            + files[BUCK]
        )
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_only_active_local_or_imported_helpers_create_providers(self):
        for imported in (False, True):
            for condition, active in (
                ("runtime.is_oss", False),
                ("not runtime.is_oss", True),
            ):
                with self.subTest(imported=imported, condition=condition):
                    files = fixture()
                    helper = "def define_schema():\n    " + library(
                        "schema", '["schema.py"]'
                    )
                    if imported:
                        files["exir/BUCK"] = (
                            'load(":targets.bzl", declare="define_schema")\n'
                        )
                        files["exir/targets.bzl"] = helper
                        call = "declare()"
                    else:
                        files["exir/BUCK"] = helper
                        call = "define_schema()"
                    files["exir/BUCK"] += (
                        library("umbrella", "[]", '[":schema"]')
                        + f"if {condition}:\n    {call}\n"
                    )
                    declarations = Declarations(Snapshot(files))
                    self.assertEqual(
                        active, declarations.get("//executorch/exir:schema") is not None
                    )
                    issues = analyze(Snapshot(files))
                    if active:
                        self.assertEqual(set(), issues)
                    else:
                        self.assertTrue(
                            any(
                                "no supported python_library provider" in issue.detail
                                for issue in issues
                            )
                        )

    def test_inactive_helpers_do_not_add_spurious_declaration_errors(self):
        for imported in (False, True):
            with self.subTest(imported=imported):
                files = fixture()
                helper = "def define_oss():\n    " + library(
                    "oss", "unsupported_sources()"
                )
                if imported:
                    files[BUCK] += 'load(":helpers.bzl", "define_oss")\n'
                    files[SCOPE + "/helpers.bzl"] = helper
                else:
                    files[BUCK] += helper
                files[BUCK] += "if runtime.is_oss:\n    define_oss()\n"
                self.assertEqual(set(), analyze(Snapshot(files)))

    def test_nested_local_helper_calls_preserve_internal_branch(self):
        files = fixture()
        files["exir/BUCK"] = (
            "def define_schema():\n    "
            + library("schema", '["schema.py"]')
            + "def define_all():\n    "
            + library("umbrella", "[]", '[":schema"]')
            + "    if not runtime.is_oss:\n        define_schema()\n"
            + "fbcode_target(_kind=define_all)\n"
        )
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_parameterized_helpers_fail_clearly_only_when_called(self):
        for imported in (False, True):
            for condition, active in (
                ("runtime.is_oss", False),
                ("not runtime.is_oss", True),
            ):
                with self.subTest(imported=imported, condition=condition):
                    before = fixture()
                    after = dict(before)
                    helper = (
                        "def declare(name, srcs):\n"
                        + "    runtime.python_library(name=name, srcs=srcs)\n"
                    )
                    if imported:
                        after[BUCK] += 'load(":helpers.bzl", "declare")\n'
                        after[SCOPE + "/helpers.bzl"] = helper
                    else:
                        after[BUCK] += helper
                    after[
                        BUCK
                    ] += f'if {condition}:\n    declare("extra", ["extra.py"])\n'
                    after[SCOPE + "/extra.py"] = (
                        "from executorch.missing import value\n"
                    )
                    issues = regressions(before, after)
                    if active:
                        self.assertTrue(
                            any(
                                issue.kind == "cannot validate"
                                and "parameterized declaration helper: declare"
                                in issue.detail
                                for issue in issues
                            )
                        )
                    else:
                        self.assertEqual(set(), issues)

    def test_internal_wrapper_selection_applies_to_helpers(self):
        for imported in (False, True):
            for wrapper, active in (
                ("fbcode_target", True),
                ("non_fbcode_target", False),
            ):
                with self.subTest(imported=imported, wrapper=wrapper):
                    files = fixture()
                    helper = "def define_extra():\n    " + library(
                        "extra", '["extra.py"]'
                    )
                    if imported:
                        files[BUCK] += 'load(":helpers.bzl", "define_extra")\n'
                        files[SCOPE + "/helpers.bzl"] = helper
                    else:
                        files[BUCK] += helper
                    files[BUCK] += f"{wrapper}(_kind=define_extra)\n"
                    declarations = Declarations(Snapshot(files))
                    self.assertEqual(
                        active,
                        declarations.get("//executorch/" + SCOPE + ":extra")
                        is not None,
                    )

    def test_parameterized_helper_definition_is_rejected_without_arguments(self):
        for imported in (False, True):
            for parameters in ('name="extra"', "*names", "**options"):
                with self.subTest(imported=imported, parameters=parameters):
                    files = fixture()
                    helper = f"def define_extra({parameters}):\n    " + library(
                        "extra", '["extra.py"]'
                    )
                    if imported:
                        files[BUCK] += 'load(":helpers.bzl", "define_extra")\n'
                        files[SCOPE + "/helpers.bzl"] = helper
                    else:
                        files[BUCK] += helper
                    files[BUCK] += "define_extra()\n"
                    self.assertTrue(
                        any(
                            issue.kind == "cannot validate"
                            and "parameterized declaration helper: define_extra"
                            in issue.detail
                            for issue in analyze(Snapshot(files))
                        )
                    )

    def test_recursive_helper_fails_clearly(self):
        files = fixture()
        files[BUCK] += "def define_extra():\n    define_extra()\ndefine_extra()\n"
        self.assertTrue(
            any(
                issue.kind == "cannot validate"
                and "recursive declaration helper: define_extra" in issue.detail
                for issue in analyze(Snapshot(files))
            )
        )

    def test_unknown_helper_condition_fails_instead_of_inventing_provider(self):
        files = fixture()
        files[BUCK] += (
            "def define_extra():\n    "
            + library("extra", '["extra.py"]')
            + 'if native.read_config("mode", "value"):\n    define_extra()\n'
        )
        issues = analyze(Snapshot(files))
        self.assertTrue(
            any(
                issue.kind == "cannot validate"
                and "conditional declaration" in issue.detail
                for issue in issues
            )
        )

    def test_unresolvable_new_inrepo_module_fails_clearly(self):
        before = fixture()
        after = dict(before)
        after[TESTER] += "from executorch.generated_unknown import value\n"
        issues = regressions(before, after)
        self.assertEqual(1, len(issues))
        self.assertEqual("cannot validate", next(iter(issues)).kind)

    def test_glob_does_not_cross_buck_package(self):
        files = fixture()
        files[BUCK] = library(
            "arm_tester_lib",
            'glob(["tester/**/*.py"])',
            '["//executorch/exir:umbrella"]',
        )
        files[SCOPE + "/tester/nested/BUCK"] = library("other", '["module.py"]')
        nested_source = SCOPE + "/tester/nested/module.py"
        files[nested_source] = "from executorch.missing import Nope\n"
        declarations = Declarations(Snapshot(files))
        root = declarations.get("//executorch/" + SCOPE + ":arm_tester_lib")
        self.assertNotIn(nested_source, root.sources)
        issues = analyze(Snapshot(files))
        self.assertEqual(1, len(issues))
        self.assertEqual(
            "//executorch/" + SCOPE + "/tester/nested:other", next(iter(issues)).owner
        )

    def test_standalone_nested_library_dependency_is_checked(self):
        for build_file in ("BUCK", "TARGETS"):
            with self.subTest(build_file=build_file):
                before = fixture()
                after = dict(before)
                package = SCOPE + "/standalone"
                source = package + "/module.py"
                after[package + "/" + build_file] = library("lib", '["module.py"]')
                after[source] = "from executorch.exir.schema import Tensor\n"
                issues = regressions(before, after)
                self.assertEqual(1, len(issues))
                issue = next(iter(issues))
                self.assertEqual("missing dependency", issue.kind)
                self.assertEqual("//executorch/" + package + ":lib", issue.owner)
                self.assertEqual(source, issue.source)
                after[package + "/" + build_file] = library(
                    "lib", '["module.py"]', '["//executorch/exir:schema"]'
                )
                self.assertEqual(set(), regressions(before, after))

    def test_nested_helper_libraries_keep_relative_package_dependencies(self):
        files = fixture()
        package = SCOPE + "/standalone/nested"
        files[package + "/BUCK"] = (
            'load(":targets.bzl", "define_libs")\ndefine_libs()\n'
        )
        files[package + "/targets.bzl"] = (
            "def define_libs():\n    "
            + library("lib", '["module.py"]', '[":schema"]')
            + "    "
            + library("schema", '["schema.py"]')
        )
        files[package + "/module.py"] = "from .schema import Tensor\n"
        files[package + "/schema.py"] = "class Tensor: pass\n"
        self.assertEqual(set(), analyze(Snapshot(files)))
        files[package + "/targets.bzl"] = files[package + "/targets.bzl"].replace(
            'deps=[":schema"]', "deps=[]"
        )
        issues = analyze(Snapshot(files))
        self.assertEqual(1, len(issues))
        self.assertIn("//executorch/" + package + ":schema", next(iter(issues)).detail)

    def test_nested_declaration_errors_identify_the_owning_file(self):
        declarations = (
            (library("lib", "choose_sources()"), "srcs: choose_sources"),
            (
                'def declare(name):\n    pass\ndeclare("lib")\n',
                "parameterized declaration helper",
            ),
            ("runtime.python_library(", "cannot read"),
        )
        for build_file in ("BUCK", "TARGETS"):
            for definition, expected in declarations:
                with self.subTest(build_file=build_file, expected=expected):
                    files = fixture()
                    filename = SCOPE + "/standalone/" + build_file
                    files[filename] = definition
                    issues = analyze(Snapshot(files))
                    relevant = [issue for issue in issues if expected in issue.detail]
                    self.assertTrue(relevant)
                    self.assertTrue(all(issue.source == filename for issue in relevant))

    def test_nested_helper_errors_identify_the_helper_file(self):
        files = fixture()
        package = SCOPE + "/standalone"
        files[package + "/TARGETS"] = (
            'load(":helpers.bzl", "define_libs")\ndefine_libs()\n'
        )
        helper = package + "/helpers.bzl"
        files[helper] = "def define_libs():\n    " + library("lib", "choose_sources()")
        issues = analyze(Snapshot(files))
        self.assertTrue(issues)
        self.assertTrue(all(issue.source == helper for issue in issues))
        files[helper] = "def define_libs():\n    declare(\n"
        issues = analyze(Snapshot(files))
        self.assertEqual(1, len(issues))
        self.assertEqual(helper, next(iter(issues)).source)

    def test_similar_sibling_package_is_outside_arm_test_scope(self):
        files = fixture()
        package = SCOPE + "_other"
        files[package + "/BUCK"] = library("lib", "choose_sources()")
        files[package + "/module.py"] = "from executorch.missing import value\n"
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_nested_baseline_gap_does_not_hide_a_different_new_gap(self):
        before = fixture()
        package = SCOPE + "/standalone"
        before[package + "/BUCK"] = library("lib", '["module.py"]')
        before[package + "/module.py"] = "from executorch.exir.schema import Tensor\n"
        self.assertEqual((set(), 1), compare(Snapshot(before), Snapshot(before)))
        after = dict(before)
        after[
            package + "/module.py"
        ] += "from executorch.devtools.pte_tool.diff_pte import PTEDiffResult\n"
        issues, unchanged = compare(Snapshot(before), Snapshot(after))
        self.assertEqual(1, unchanged)
        self.assertEqual(1, len(issues))
        self.assertIn("diff_pte_lib", next(iter(issues)).detail)

    def test_deleted_nested_owner_with_retained_source_fails(self):
        before = fixture()
        package = SCOPE + "/standalone"
        source = package + "/module.py"
        before[package + "/TARGETS"] = library("lib", '["module.py"]')
        before[source] = "value = 42\n"
        after = dict(before)
        del after[package + "/TARGETS"]
        issues = regressions(before, after)
        self.assertEqual(1, len(issues))
        issue = next(iter(issues))
        self.assertEqual("ownership", issue.category)
        self.assertEqual(source, issue.source)
        del after[source]
        self.assertEqual(set(), regressions(before, after))

    def test_loop_generated_library_is_not_silently_ignored(self):
        before = fixture()
        after = dict(before)
        after[
            BUCK
        ] += 'for name in ["extra"]:\n    runtime.python_library(name=name, srcs=["extra.py"])\n'
        after[SCOPE + "/extra.py"] = "from executorch.missing import value\n"
        issues = regressions(before, after)
        self.assertTrue(
            any(
                i.kind == "cannot validate"
                and "unsupported declaration loop" in i.detail
                for i in issues
            )
        )

    def test_known_pytest_generation_loop_is_handled_by_registration(self):
        files = fixture()
        files[
            BUCK
        ] += 'load(":targets.bzl", "define_arm_tests")\nfbcode_target(_kind=define_arm_tests)\n'
        files[
            TARGETS
        ] += "    for test_file in test_files:\n        python_pytest(name=test_file, srcs=[test_file])\n"
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_oss_only_test_registration_is_valid(self):
        before = fixture()
        after = dict(before)
        after[SCOPE + "/misc/test_oss.py"] = "def test_oss():\n    pass\n"
        after[
            TARGETS
        ] += '    if runtime.is_oss:\n        test_files += ["misc/test_oss.py"]\n'
        self.assertEqual(set(), regressions(before, after))

    def test_provider_suggestion_change_is_not_new_violation(self):
        before = fixture()
        before[BUCK] = library("arm_tester_lib", 'glob(["tester/*.py"])')
        after = dict(before)
        after["exir/BUCK"] += library("schema_alternative", '["schema.py"]')
        self.assertNotEqual(
            {i.detail for i in analyze(Snapshot(before))},
            {i.detail for i in analyze(Snapshot(after))},
        )
        self.assertEqual(set(), regressions(before, after))

    def test_diagnostic_category_change_only_adds_external_warnings(self):
        before = fixture()
        before[BUCK] = library("arm_tester_lib", 'glob(["tester/*.py"])')
        after = dict(before)
        after[BUCK] = library(
            "arm_tester_lib",
            'glob(["tester/*.py"])',
            '["//executorch/internal:unavailable"]',
        )
        issues = regressions(before, after)
        self.assertTrue(issues)
        self.assertTrue(all(i.severity == "warning" for i in issues))

    def test_lazy_external_package_errors_keep_the_original_cause(self):
        declarations = (
            (
                'for name in ["schema"]:\n'
                '    runtime.python_library(name=name, srcs=["schema.py"])\n',
                "unsupported declaration loop",
            ),
            (
                "def declare(name):\n"
                '    runtime.python_library(name=name, srcs=["schema.py"])\n'
                'declare("schema")\n',
                "parameterized",
            ),
            (
                "if configured():\n"
                '    runtime.python_library(name="schema", srcs=["schema.py"])\n',
                "conditional",
            ),
        )
        for declaration, cause in declarations:
            for through_dependency in (False, True):
                with self.subTest(cause=cause, through_dependency=through_dependency):
                    files = fixture()
                    files["exir/BUCK"] = declaration
                    files[BUCK] = library(
                        "arm_tester_lib",
                        '["tester/runner.py"]',
                        '["//executorch/exir:schema"]' if through_dependency else "[]",
                    )
                    issues = analyze(Snapshot(files))
                    self.assertTrue(all(i.severity == "warning" for i in issues))
                    self.assertTrue(
                        any(
                            i.source == "exir/BUCK" and cause in i.detail
                            for i in issues
                        ),
                        issues,
                    )

    def test_lazy_external_helper_error_names_helper_file(self):
        files = fixture()
        files["exir/BUCK"] = 'load(":rules.bzl", "declare")\ndeclare()\n'
        files["exir/rules.bzl"] = (
            "def declare():\n"
            "    if configured():\n"
            '        runtime.python_library(name="schema", srcs=["schema.py"])\n'
        )
        issues = analyze(Snapshot(files))
        self.assertTrue(
            any(
                i.source == "exir/rules.bzl"
                and "conditional" in i.detail
                and i.severity == "warning"
                for i in issues
            ),
            issues,
        )
        self.assertTrue(all(i.severity == "warning" for i in issues))

    def test_computed_external_sources_report_the_original_expression(self):
        for through_dependency in (False, True):
            with self.subTest(through_dependency=through_dependency):
                files = fixture()
                files["exir/BUCK"] = library("schema", "choose_sources()")
                files[BUCK] = library(
                    "arm_tester_lib",
                    '["tester/runner.py"]',
                    '["//executorch/exir:schema"]' if through_dependency else "[]",
                )
                issues = analyze(Snapshot(files))
                self.assertTrue(all(i.severity == "warning" for i in issues))
                self.assertTrue(
                    any(
                        i.source == "exir/BUCK" and "choose_sources()" in i.detail
                        for i in issues
                    ),
                    issues,
                )

    def test_computed_external_dependencies_do_not_cascade_to_errors(self):
        before = fixture()
        after = dict(before)
        after["exir/BUCK"] = library("umbrella", "[]", "choose_deps()") + library(
            "schema", '["schema.py"]'
        )
        issues = regressions(before, after)
        self.assertTrue(issues)
        self.assertTrue(all(i.severity == "warning" for i in issues))
        self.assertTrue(
            any(i.source == "exir/BUCK" and "choose_deps()" in i.detail for i in issues)
        )
        self.assertTrue(
            any(
                i.source == TESTER and "exir/BUCK: deps: choose_deps()" in i.detail
                for i in issues
            )
        )

    def test_known_missing_dependency_still_fails_with_external_package_warning(self):
        files = fixture()
        files[BUCK] = library("arm_tester_lib", '["tester/runner.py"]')
        files["exir/BUCK"] += (
            'for name in ["other"]:\n'
            "    runtime.python_library(name=name, srcs=[])\n"
        )
        issues = analyze(Snapshot(files))
        self.assertTrue(
            any(
                i.kind == "missing dependency" and i.severity == "error" for i in issues
            )
        )
        self.assertTrue(
            any(i.source == "exir/BUCK" and i.severity == "warning" for i in issues)
        )

    def test_unsupported_arm_provider_outside_test_scope_still_fails(self):
        files = fixture()
        files[TESTER] = "from executorch.backends.arm.provider import value\n"
        files["backends/arm/BUCK"] = library("provider", "choose_sources()")
        files["backends/arm/provider.py"] = "value = 1\n"
        issues = analyze(Snapshot(files))
        self.assertTrue(
            any(
                i.source == "backends/arm/BUCK"
                and "choose_sources()" in i.detail
                and i.severity == "error"
                for i in issues
            )
        )

    def test_unrelated_external_packages_are_not_reported(self):
        files = fixture()
        files["unrelated/BUCK"] = (
            'for name in ["other"]:\n'
            "    runtime.python_library(name=name, srcs=[])\n"
        )
        self.assertEqual(set(), analyze(Snapshot(files)))

    def test_unchanged_external_warnings_are_baselined(self):
        files = fixture()
        files["exir/BUCK"] = library("umbrella", "[]", "choose_deps()") + library(
            "schema", '["schema.py"]'
        )
        new, unchanged = compare(Snapshot(files), Snapshot(files))
        self.assertEqual(set(), new)
        self.assertGreater(unchanged, 0)

    def test_warning_becoming_proven_missing_dependency_is_new_error(self):
        before = fixture()
        before["exir/BUCK"] = library("umbrella", "[]", "choose_deps()") + library(
            "schema", '["schema.py"]'
        )
        after = dict(before)
        after["exir/BUCK"] = library("umbrella", "[]") + library(
            "schema", '["schema.py"]'
        )
        issues = regressions(before, after)
        self.assertEqual(1, len(issues))
        issue = next(iter(issues))
        self.assertEqual("missing dependency", issue.kind)
        self.assertEqual("error", issue.severity)
        self.assertEqual("executorch.exir.schema", issue.subject)

    def test_cli_reports_warnings_without_failing_and_still_fails_errors(self):
        before = fixture()
        for add_arm_error in (False, True):
            with self.subTest(add_arm_error=add_arm_error):
                after = dict(before)
                after["exir/BUCK"] = library(
                    "umbrella", "[]", "choose_deps()"
                ) + library("schema", '["schema.py"]')
                if add_arm_error:
                    after[SCOPE + "/misc/test_new.py"] = "def test_new(): pass\n"
                output = io.StringIO()
                with mock.patch.object(
                    checker,
                    "GitSnapshot",
                    side_effect=[Snapshot(before), Snapshot(after)],
                ), contextlib.redirect_stdout(output):
                    status = checker.main(["--repo", ".", "--base", "base"])
                self.assertEqual(1 if add_arm_error else 0, status)
                report = output.getvalue()
                self.assertIn("FAIL" if add_arm_error else "PASS", report)
                self.assertIn(
                    "New errors: 1" if add_arm_error else "New errors: 0", report
                )
                self.assertIn("New warnings: 2", report)
                self.assertIn("WARNING: Cannot validate", report)
                self.assertIn("File: exir/BUCK", report)
                self.assertIn("choose_deps()", report)
                if add_arm_error:
                    self.assertIn("ERROR: Unregistered test", report)

    def test_nested_explicit_python_test_registration_is_valid(self):
        for kind in ("python_test", "python_pytest", "python_unittest"):
            with self.subTest(kind=kind):
                before = fixture()
                after = dict(before)
                after[SCOPE + "/unit/test_unit.py"] = "def test_unit():\n    pass\n"
                after[SCOPE + "/unit/BUCK"] = (
                    f'runtime.{kind}(name="test_unit", srcs=["test_unit.py"], deps=[])\n'
                )
                self.assertEqual(set(), regressions(before, after))

    def test_deleted_owner_with_retained_source_fails(self):
        before = fixture()
        for retain_aggregator in (False, True):
            with self.subTest(retain_aggregator=retain_aggregator):
                after = dict(before)
                after[BUCK] = (
                    library("arm_tester", "[]", '[":arm_tester_lib"]')
                    if retain_aggregator
                    else ""
                )
                issues = regressions(before, after)
                self.assertTrue(
                    any(
                        i.category == "ownership" and i.source == TESTER for i in issues
                    )
                )
                if retain_aggregator:
                    self.assertTrue(
                        any("unresolved scoped dependency" in i.detail for i in issues)
                    )

    def test_deleted_owner_sources_and_edges_is_valid(self):
        before = fixture()
        before[BUCK] += library("arm_tester", "[]", '[":arm_tester_lib"]')
        after = dict(before)
        after[BUCK] = ""
        del after[TESTER]
        self.assertEqual(set(), regressions(before, after))

    def test_library_rename_with_retained_sources_is_valid(self):
        before = fixture()
        after = dict(before)
        after[BUCK] = after[BUCK].replace("arm_tester_lib", "renamed_lib")
        self.assertEqual(set(), regressions(before, after))


if __name__ == "__main__":
    unittest.main()
