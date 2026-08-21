# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
import zipfile

import torch

from executorch.exir._serialize._ptn import (
    ALIASES_ENTRY,
    PTG_ENTRY,
    read_ptn,
    SAFETENSORS_ENTRY,
    write_ptn,
)
from safetensors.torch import load as safetensors_load

# write_ptn does not interpret the graph blob, so a stand-in keeps this a unit test.
_PTG = b"\x00\x00\x00\x00NPTG stand-in graph blob"


def _entries(path: str) -> set[str]:
    with zipfile.ZipFile(path) as pkg:
        return set(pkg.namelist())


def _read_entry(path: str, entry: str) -> bytes:
    with zipfile.ZipFile(path) as pkg:
        return pkg.read(entry)


def _rewrite(path: str, overrides: dict[str, bytes]) -> None:
    """Rebuild a package in place, replacing or adding the named entries."""
    with zipfile.ZipFile(path) as src:
        entries = {name: src.read(name) for name in src.namelist()}
    entries.update(overrides)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as pkg:
        for name, data in entries.items():
            pkg.writestr(name, data)


class WritePtnTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "m.ptn")

    def test_no_constants_omits_safetensors(self):
        write_ptn(self.path, _PTG, {})

        self.assertEqual(_entries(self.path), {PTG_ENTRY})
        self.assertEqual(read_ptn(self.path), (_PTG, {}))

    def test_distinct_constants_omit_alias_map(self):
        constants = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
        write_ptn(self.path, _PTG, constants)

        self.assertEqual(_entries(self.path), {PTG_ENTRY, SAFETENSORS_ENTRY})
        ptg, out = read_ptn(self.path)
        self.assertEqual(ptg, _PTG)
        self.assertEqual(set(out), {"a", "b"})
        for key, tensor in constants.items():
            self.assertTrue(torch.equal(out[key], tensor))

    def test_byte_identical_constants_dedup(self):
        shared = torch.tensor([1.0, 2.0, 3.0])
        write_ptn(
            self.path,
            _PTG,
            {"a": shared, "b": shared.clone(), "c": torch.tensor([9.0])},
        )

        # Keys are packed in sorted order, so "a" owns the entry and "b" aliases.
        self.assertEqual(json.loads(_read_entry(self.path, ALIASES_ENTRY)), {"b": "a"})
        owners = safetensors_load(_read_entry(self.path, SAFETENSORS_ENTRY))
        self.assertEqual(set(owners), {"a", "c"})

        _, out = read_ptn(self.path)
        self.assertEqual(set(out), {"a", "b", "c"})
        self.assertTrue(torch.equal(out["b"], shared))

    def test_mutable_key_does_not_dedup_with_identical_immutable_key(self):
        zeros = torch.zeros(4)
        write_ptn(
            self.path,
            _PTG,
            {"immutable": zeros, "mutable": zeros.clone()},
            mutable_keys=frozenset({"mutable"}),
        )

        self.assertNotIn(ALIASES_ENTRY, _entries(self.path))
        owners = safetensors_load(_read_entry(self.path, SAFETENSORS_ENTRY))
        self.assertEqual(set(owners), {"immutable", "mutable"})

    def test_shared_mutable_storage_is_rejected(self):
        base = torch.arange(8, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "share source storage"):
            write_ptn(
                self.path,
                _PTG,
                {"a": base[:4], "b": base[2:6]},
                mutable_keys=frozenset({"a", "b"}),
            )

    def test_unknown_mutable_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no tensor data"):
            write_ptn(
                self.path,
                _PTG,
                {"weight": torch.ones(2)},
                mutable_keys=frozenset({"missing"}),
            )

    def test_safetensors_metadata_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved by safetensors"):
            write_ptn(
                self.path,
                _PTG,
                {"__metadata__": torch.ones(2)},
            )

    def test_data_key_must_be_a_string(self):
        with self.assertRaisesRegex(TypeError, "data key must be str"):
            write_ptn(self.path, _PTG, {1: torch.ones(2)})

    def test_data_value_must_be_a_tensor(self):
        with self.assertRaisesRegex(TypeError, "must be a Tensor"):
            write_ptn(self.path, _PTG, {"weight": object()})

    def test_unsupported_dtype_fails_before_replacing_existing_package(self):
        write_ptn(self.path, _PTG, {"a": torch.ones(2)})
        with open(self.path, "rb") as artifact:
            before = artifact.read()

        with self.assertRaisesRegex(KeyError, "complex128"):
            write_ptn(self.path, _PTG, {"a": torch.ones(2, dtype=torch.complex128)})

        with open(self.path, "rb") as artifact:
            self.assertEqual(artifact.read(), before)

    def test_tied_weight_sharing_storage_round_trips(self):
        # The same tensor under two keys dedups to one owner.
        base = torch.arange(6, dtype=torch.float32)
        write_ptn(self.path, _PTG, {"w": base, "tied": base})

        owners = safetensors_load(_read_entry(self.path, SAFETENSORS_ENTRY))
        self.assertEqual(set(owners), {"tied"})
        _, out = read_ptn(self.path)
        self.assertEqual(set(out), {"w", "tied"})
        self.assertTrue(torch.equal(out["w"], base))
        self.assertTrue(torch.equal(out["tied"], base))

    def test_reshaped_view_of_whole_storage(self):
        # base and base.view(2, 3) both cover the same storage end to end, and
        # their differing shapes stop content dedup collapsing them, so both are
        # owners over one buffer. Writing each owner's bytes independently is what
        # keeps that valid; safetensors.save_file would reject it.
        base = torch.arange(6, dtype=torch.float32)
        write_ptn(self.path, _PTG, {"flat": base, "matrix": base.view(2, 3)})

        owners = safetensors_load(_read_entry(self.path, SAFETENSORS_ENTRY))
        self.assertEqual(set(owners), {"flat", "matrix"})
        _, out = read_ptn(self.path)
        self.assertEqual(out["flat"].shape, base.shape)
        self.assertEqual(tuple(out["matrix"].shape), (2, 3))
        self.assertTrue(torch.equal(out["flat"], base))
        self.assertTrue(torch.equal(out["matrix"], base.view(2, 3)))

    def test_overlapping_views_of_one_storage(self):
        # Different content, one storage: content dedup cannot collapse these.
        base = torch.arange(8, dtype=torch.float32)
        write_ptn(self.path, _PTG, {"lo": base[0:4], "hi": base[2:6]})

        owners = safetensors_load(_read_entry(self.path, SAFETENSORS_ENTRY))
        self.assertEqual(set(owners), {"lo", "hi"})
        _, out = read_ptn(self.path)
        self.assertTrue(torch.equal(out["lo"], base[0:4]))
        self.assertTrue(torch.equal(out["hi"], base[2:6]))

    def test_narrower_view_of_one_storage(self):
        base = torch.arange(8, dtype=torch.float32)
        write_ptn(self.path, _PTG, {"tail": base[4:]})

        _, out = read_ptn(self.path)
        self.assertTrue(torch.equal(out["tail"], base[4:]))

    def test_same_content_different_dtype_not_deduped(self):
        write_ptn(
            self.path,
            _PTG,
            {
                "f": torch.ones(4, dtype=torch.float32),
                "i": torch.ones(4, dtype=torch.int32),
            },
        )

        self.assertNotIn(ALIASES_ENTRY, _entries(self.path))
        _, out = read_ptn(self.path)
        self.assertEqual(out["f"].dtype, torch.float32)
        self.assertEqual(out["i"].dtype, torch.int32)

    def test_dtypes_round_trip(self):
        # Also validates the hand-written header against the real reader for every
        # dtype code, including ones numpy cannot express.
        dtypes = [
            torch.float64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
            torch.complex64,
        ]
        for name in ("uint16", "uint32", "uint64"):
            dtype = getattr(torch, name, None)
            if dtype is not None:
                dtypes.append(dtype)

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.ones((2, 3), dtype=dtype)
                path = os.path.join(
                    self._tmp.name, f"{str(dtype).rsplit('.', 1)[-1]}.ptn"
                )
                write_ptn(path, _PTG, {"t": tensor})
                _, out = read_ptn(path)
                self.assertEqual(out["t"].dtype, dtype)
                self.assertEqual(out["t"].shape, tensor.shape)
                self.assertTrue(torch.equal(out["t"], tensor))

    def test_multiple_tensors_read_back_at_correct_offsets(self):
        # Guards the hand-written data_offsets: differing element sizes mean a
        # mistake shifts later tensors rather than merely corrupting one.
        constants = {
            "a": torch.arange(3, dtype=torch.float32),
            "b": torch.arange(5, dtype=torch.int64),
            "c": torch.arange(7, dtype=torch.int8),
        }
        write_ptn(self.path, _PTG, constants)

        _, out = read_ptn(self.path)
        for key, tensor in constants.items():
            self.assertTrue(torch.equal(out[key], tensor), f"{key} mismatched")

    def test_member_names_are_fixed_so_renaming_is_safe(self):
        write_ptn(self.path, _PTG, {"a": torch.zeros(2)})
        self.assertEqual(_entries(self.path), {PTG_ENTRY, SAFETENSORS_ENTRY})

        renamed = os.path.join(self._tmp.name, "renamed-artifact.ptn")
        os.rename(self.path, renamed)
        ptg, out = read_ptn(renamed)
        self.assertEqual(ptg, _PTG)
        self.assertTrue(torch.equal(out["a"], torch.zeros(2)))

    def test_creates_missing_parent_directory(self):
        path = os.path.join(self._tmp.name, "nested", "deeper", "m.ptn")
        write_ptn(path, _PTG, {})

        self.assertTrue(os.path.isfile(path))

    def test_alias_naming_unknown_owner_raises(self):
        write_ptn(self.path, _PTG, {"a": torch.zeros(2)})
        _rewrite(self.path, {ALIASES_ENTRY: json.dumps({"b": "nope"}).encode()})

        with self.assertRaisesRegex(ValueError, "has no safetensors entry"):
            read_ptn(self.path)

    def test_aliases_must_be_a_json_object(self):
        write_ptn(self.path, _PTG, {"a": torch.zeros(2)})
        _rewrite(self.path, {ALIASES_ENTRY: b"[]"})

        with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
            read_ptn(self.path)

    def test_alias_owner_must_be_a_string(self):
        write_ptn(self.path, _PTG, {"a": torch.zeros(2)})
        _rewrite(self.path, {ALIASES_ENTRY: json.dumps({"b": 1}).encode()})

        with self.assertRaisesRegex(ValueError, "owner name must be a string"):
            read_ptn(self.path)

    def test_key_both_owner_and_alias_raises(self):
        write_ptn(self.path, _PTG, {"a": torch.zeros(2), "b": torch.ones(2)})
        _rewrite(self.path, {ALIASES_ENTRY: json.dumps({"a": "b"}).encode()})

        with self.assertRaisesRegex(
            ValueError, "both a safetensors owner and an alias"
        ):
            read_ptn(self.path)
