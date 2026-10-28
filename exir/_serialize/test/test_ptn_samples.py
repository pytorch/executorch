# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock

import torch
from executorch.exir._serialize._ptn import read_ptn, write_ptn
from executorch.exir._serialize._ptn_samples import (
    Call,
    read_samples,
    Sample,
    SampleSet,
    write_samples,
)

_PTG = b"\x00\x00\x00\x00NPTG stand-in graph blob"
_INDEX = "samples/golden/index.json"
_DATA = "samples/golden/data.safetensors"


def _golden() -> SampleSet:
    return SampleSet(
        samples=[
            Sample(
                [
                    Call(
                        "forward",
                        inputs=[torch.randn(1, 3)],
                        outputs=[torch.randn(1, 2)],
                    )
                ]
            ),
            Sample(
                [
                    Call(
                        "forward",
                        inputs=[torch.randn(1, 3)],
                        outputs=[torch.randn(1, 2)],
                    )
                ]
            ),
        ],
        description="two calls",
        source="eager/cpu/f32",
        tol=(1e-4, 1e-4),
    )


def _members(path: str) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as pkg:
        return {name: pkg.read(name) for name in pkg.namelist()}


def _write_aligned(pkg: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Store `data` so it starts 64-byte aligned, padding the local header."""
    start = pkg.fp.tell() + 30 + len(name.encode()) + 6
    padding = (-start) % 64
    zinfo = zipfile.ZipInfo(name)
    zinfo.extra = struct.pack("<HHH", 0xD935, 2 + padding, 64) + bytes(padding)
    pkg.writestr(zinfo, data)


def _rewrite(path: str, overrides: dict[str, bytes | None]) -> None:
    """Rebuild a package, replacing, adding or (for None) dropping entries."""
    entries = _members(path)
    entries.update(overrides)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as pkg:
        for name, data in entries.items():
            if data is None:
                continue
            if name.endswith(".safetensors"):
                _write_aligned(pkg, name, data)
            else:
                pkg.writestr(name, data)


def _data_start(path: str) -> int:
    """File offset where the golden set's tensor data begins."""
    with zipfile.ZipFile(path) as pkg:
        info = pkg.getinfo(_DATA)
        data = pkg.read(_DATA)
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        name_size, extra_size = struct.unpack("<HH", f.read(30)[26:30])
    header_size = int.from_bytes(data[:8], "little")
    return info.header_offset + 30 + name_size + extra_size + 8 + header_size


def _edit_index(path: str, edit) -> None:
    index = json.loads(_members(path)[_INDEX])
    edit(index)
    _rewrite(path, {_INDEX: json.dumps(index).encode()})


class PtnSamplesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "m.ptn")
        write_ptn(self.path, _PTG, {"w": torch.ones(4)})

    def assertValuesEqual(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assertValuesEqual(a, e)
        else:
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(actual, expected)

    def assertSetsEqual(self, actual: SampleSet, expected: SampleSet):
        self.assertEqual(actual.description, expected.description)
        self.assertEqual(actual.source, expected.source)
        self.assertEqual(actual.tol, expected.tol)
        self.assertEqual(len(actual.samples), len(expected.samples))
        for a, e in zip(actual.samples, expected.samples):
            self.assertEqual(a.tol, e.tol)
            self.assertEqual(len(a.calls), len(e.calls))
            for ac, ec in zip(a.calls, e.calls):
                self.assertCallsEqual(ac, ec)

    def assertCallsEqual(self, a: Call, e: Call):
        self.assertEqual(a.method, e.method)
        self.assertValuesEqual(a.inputs, e.inputs)
        self.assertEqual(a.outputs is None, e.outputs is None)
        if e.outputs is not None:
            self.assertValuesEqual(a.outputs, e.outputs)
        self.assertEqual(a.intermediates is None, e.intermediates is None)
        if e.intermediates is not None:
            self.assertEqual(a.intermediates.keys(), e.intermediates.keys())
            for name in e.intermediates:
                self.assertValuesEqual(a.intermediates[name], e.intermediates[name])
        self.assertEqual(a.tol, e.tol)

    def test_package_without_samples_reads_empty(self):
        self.assertEqual(read_samples(self.path), {})

    def test_golden_round_trips(self):
        golden = _golden()
        write_samples(self.path, "golden", golden)

        sets = read_samples(self.path)
        self.assertEqual(list(sets), ["golden"])
        self.assertSetsEqual(sets["golden"], golden)

    def test_intermediates_and_nested_tol_round_trip(self):
        debug = SampleSet(
            samples=[
                Sample(
                    [
                        Call(
                            "forward",
                            inputs=[torch.randn(2, 4)],
                            outputs=[torch.randn(2, 4)],
                            intermediates={
                                "linear": torch.randn(2, 8),
                                "gelu_1": torch.randn(2, 8),
                            },
                            tol=(1e-3, 5e-4),
                        )
                    ],
                    tol=(5e-4, 5e-4),
                )
            ],
            tol=(1e-4, 1e-4),
        )
        write_samples(self.path, "debug", debug)

        self.assertSetsEqual(read_samples(self.path)["debug"], debug)

    def test_calls_round_trip_in_order(self):
        prefill = Call(
            "forward",
            [torch.arange(6).view(1, 6), torch.arange(6)],
            [torch.randn(1, 1, 5)],
        )
        decode = [
            Call(
                "forward",
                [torch.tensor([[t]]), torch.tensor([6 + i])],
                [torch.randn(1, 1, 5)],
            )
            for i, t in enumerate((11, 12, 13))
        ]
        golden = SampleSet([Sample([prefill, *decode]), Sample([prefill])])
        write_samples(self.path, "golden", golden)

        self.assertSetsEqual(read_samples(self.path)["golden"], golden)

    def test_inputs_only_round_trips(self):
        calib = SampleSet(
            samples=[Sample([Call("forward", [torch.randn(3)])]) for _ in range(3)]
        )
        write_samples(self.path, "calibration", calib)

        self.assertSetsEqual(read_samples(self.path)["calibration"], calib)

    def test_scalars_lists_and_dtypes_round_trip(self):
        mixed = SampleSet(
            samples=[
                Sample(
                    [
                        Call(
                            "forward",
                            inputs=[
                                torch.arange(6, dtype=torch.int64).reshape(2, 3),
                                [
                                    torch.ones(2, dtype=torch.bfloat16),
                                    torch.zeros(1, dtype=torch.bool),
                                ],
                                3,
                                2.0,
                                True,
                            ],
                            outputs=[torch.ones(2, dtype=torch.float16), 7],
                        )
                    ]
                )
            ]
        )
        write_samples(self.path, "mixed", mixed)

        self.assertSetsEqual(read_samples(self.path)["mixed"], mixed)

    def test_samples_may_differ_in_shape(self):
        dynamic = SampleSet(
            samples=[Sample([Call("forward", [torch.randn(1, n, 8)])]) for n in (3, 9)]
        )
        write_samples(self.path, "dynamic", dynamic)

        samples = read_samples(self.path)["dynamic"].samples
        self.assertEqual([s.calls[0].inputs[0].shape[1] for s in samples], [3, 9])

    def test_keys_are_json_paths(self):
        write_samples(
            self.path,
            "golden",
            SampleSet(
                samples=[
                    Sample(
                        [
                            Call(
                                "forward",
                                inputs=[torch.ones(1), [torch.ones(1), 2]],
                                outputs=[torch.ones(1)],
                                intermediates={"linear": torch.ones(1)},
                            ),
                            Call("forward", inputs=[torch.ones(1)]),
                        ]
                    )
                ]
            ),
        )

        calls = json.loads(_members(self.path)[_INDEX])["samples"][0]["calls"]
        at = "samples[0].calls[0]"
        self.assertEqual(calls[0]["in"], [f"{at}.in[0]", [f"{at}.in[1][0]", 2]])
        self.assertEqual(calls[0]["out"], [f"{at}.out[0]"])
        self.assertEqual(
            calls[0]["intermediates"], {"linear": f"{at}.intermediates.linear"}
        )
        self.assertEqual(
            calls[1], {"method": "forward", "in": ["samples[0].calls[1].in[0]"]}
        )

    def test_sample_spans_methods(self):
        prefill = Call("prefill", [torch.arange(6).view(1, 6)], [torch.randn(1, 6, 5)])
        decode = Call("decode", [torch.tensor([[11]])], [torch.randn(1, 1, 5)])
        golden = SampleSet([Sample([prefill, decode])])
        write_samples(self.path, "golden", golden)

        calls = json.loads(_members(self.path)[_INDEX])["samples"][0]["calls"]
        self.assertEqual([c["method"] for c in calls], ["prefill", "decode"])
        self.assertSetsEqual(read_samples(self.path)["golden"], golden)

    def test_sets_coexist(self):
        write_samples(self.path, "golden", _golden())
        bench = SampleSet([Sample([Call("forward", [torch.ones(1)])])])
        write_samples(self.path, "bench", bench)

        self.assertEqual(sorted(read_samples(self.path)), ["bench", "golden"])

    def test_append_leaves_program_untouched(self):
        before = _members(self.path)
        write_samples(self.path, "golden", _golden())

        after = _members(self.path)
        for name, data in before.items():
            self.assertEqual(after[name], data, name)
        ptg, constants = read_ptn(self.path)
        self.assertEqual(ptg, _PTG)
        self.assertTrue(torch.equal(constants["w"], torch.ones(4)))

    def test_append_through_symlink_updates_target(self):
        link = os.path.join(self._tmp.name, "link.ptn")
        os.symlink(self.path, link)
        write_samples(link, "golden", _golden())

        self.assertTrue(os.path.islink(link))
        self.assertEqual(list(read_samples(self.path)), ["golden"])

    def test_members_are_stored_and_data_aligned(self):
        # Vary the preceding index.json size so the data member starts at every
        # offset modulo the alignment.
        pad = SampleSet([Sample([Call("forward", [torch.ones(1)])])])
        for size in range(64):
            with self.subTest(size=size):
                write_ptn(self.path, _PTG, {"w": torch.ones(4)})
                pad.description = "x" * size
                write_samples(self.path, "pad", pad)
                write_samples(self.path, "golden", _golden())
                with zipfile.ZipFile(self.path) as pkg:
                    for name in (_INDEX, _DATA):
                        self.assertEqual(
                            pkg.getinfo(name).compress_type, zipfile.ZIP_STORED
                        )
                self.assertEqual(_data_start(self.path) % 64, 0)
                read_samples(self.path)

    def test_reader_rejects_misaligned_data(self):
        # Program sizes one byte apart cannot both leave the data aligned.
        for size in (0, 1):
            write_ptn(self.path, b"x" * size, {})
            write_samples(self.path, "golden", _golden())
            entries = _members(self.path)
            with zipfile.ZipFile(self.path, "w", zipfile.ZIP_STORED) as pkg:
                for name, data in entries.items():
                    pkg.writestr(name, data)
            if _data_start(self.path) % 64:
                break
        with self.assertRaisesRegex(ValueError, "not 64-byte aligned"):
            read_samples(self.path)

    def test_failed_append_leaves_package_untouched(self):
        write_samples(self.path, "golden", _golden())
        with open(self.path, "rb") as f:
            before = f.read()
        with mock.patch(
            "executorch.exir._serialize._ptn_samples._write_safetensors",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                write_samples(self.path, "other", _golden())

        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.listdir(self._tmp.name), ["m.ptn"])

    def test_writer_rejects_existing_set(self):
        write_samples(self.path, "golden", _golden())
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_samples(self.path, "golden", _golden())

    def test_writer_rejects_bad_names(self):
        for name in ("gol/den", "gol.den", ""):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "must match"):
                    write_samples(self.path, name, _golden())
        no_method = SampleSet([Sample([Call("", [torch.ones(1)])])])
        with self.assertRaisesRegex(ValueError, "method must be a non-empty string"):
            write_samples(self.path, "golden", no_method)
        bad = SampleSet(
            [
                Sample(
                    [
                        Call(
                            "forward",
                            [torch.ones(1)],
                            intermediates={"a.b": torch.ones(1)},
                        )
                    ]
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            write_samples(self.path, "debug", bad)

    def test_writer_rejects_bad_values(self):
        cases = [
            (SampleSet([]), ValueError, "at least one sample"),
            (SampleSet([Sample([])]), ValueError, "at least one call"),
            (
                SampleSet([Sample([Call("forward", [float("nan")])])]),
                ValueError,
                "not finite",
            ),
            (
                SampleSet(
                    [Sample([Call("forward", [torch.ones(1)])])], tol=(-1.0, 0.0)
                ),
                ValueError,
                "tol",
            ),
            (
                SampleSet(
                    [Sample([Call("forward", [torch.ones(1)], tol=(0.0, -1.0))])]
                ),
                ValueError,
                r"samples\[0\]\.calls\[0\]: tol",
            ),
            (
                SampleSet([Sample([Call("forward", ["x"])])]),
                TypeError,
                "unsupported value",
            ),
            (
                SampleSet([Sample([Call("forward", [torch.ones(1)])])], description=1),
                TypeError,
                "description must be a string",
            ),
            (
                SampleSet([Sample([Call("forward", torch.ones(1))])]),
                TypeError,
                "inputs must be a list",
            ),
            (
                SampleSet([Sample([Call("forward", [torch.ones(1)], torch.ones(1))])]),
                TypeError,
                "outputs must be a list",
            ),
            (
                SampleSet(
                    [Sample([Call("forward", [torch.ones(1, dtype=torch.complex128)])])]
                ),
                KeyError,
                "cannot represent dtype",
            ),
        ]
        for samples, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    write_samples(self.path, "golden", samples)
        self.assertEqual(read_samples(self.path), {})

    def test_writer_warns_on_loose_tol(self):
        one = [torch.ones(1)]
        cases = [
            (SampleSet([Sample([Call("forward", one)])], tol=(2.0, 0.05)), "set"),
            (
                SampleSet([Sample([Call("forward", one)], tol=(0.0, 2e-3))]),
                r"samples\[0\]",
            ),
            (
                SampleSet(
                    [
                        Sample(
                            [
                                Call("forward", one),
                                Call("forward", one, tol=(2e-3, 0.0)),
                            ]
                        )
                    ]
                ),
                r"samples\[0\]\.calls\[1\]",
            ),
        ]
        for i, (loose, where) in enumerate(cases):
            with self.assertWarnsRegex(
                UserWarning, f"{where} tol .* above 0.001"
            ) as caught:
                write_samples(self.path, f"loose{i}", loose)
            self.assertEqual(caught.filename, __file__)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            write_samples(self.path, "golden", _golden())
            tight = SampleSet(
                [Sample([Call("forward", [torch.ones(1)], tol=(1e-3, 1e-3))])]
            )
            write_samples(self.path, "tight", tight)

    def test_writer_requires_a_program(self):
        path = os.path.join(self._tmp.name, "empty.ptn")
        with zipfile.ZipFile(path, "w") as pkg:
            pkg.writestr("other", b"")
        with self.assertRaisesRegex(ValueError, "has no program.ptg"):
            write_samples(path, "golden", _golden())

    def test_reader_rejects_bad_layout(self):
        cases = [
            ({_DATA: None}, "missing data.safetensors"),
            ({_INDEX: None}, "missing index.json"),
            ({"samples/golden/notes.txt": b""}, "unexpected member"),
            ({"samples/stray.json": b""}, "unexpected member"),
            ({"samples/gol.den/index.json": b""}, "unexpected member"),
            ({"samples/golden/x/index.json": b""}, "unexpected member"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                write_ptn(self.path, _PTG, {})
                write_samples(self.path, "golden", _golden())
                _rewrite(self.path, overrides)
                with self.assertRaisesRegex(ValueError, message):
                    read_samples(self.path)

    def test_reader_rejects_compressed_member(self):
        write_samples(self.path, "golden", _golden())
        entries = _members(self.path)
        with zipfile.ZipFile(self.path, "w") as pkg:
            for name, data in entries.items():
                compression = (
                    zipfile.ZIP_DEFLATED if name == _INDEX else zipfile.ZIP_STORED
                )
                pkg.writestr(name, data, compress_type=compression)
        with self.assertRaisesRegex(ValueError, "is compressed"):
            read_samples(self.path)

    def test_reader_rejects_duplicate_member(self):
        write_samples(self.path, "golden", _golden())
        index = _members(self.path)[_INDEX]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(self.path, "a") as pkg:
                pkg.writestr(_INDEX, index)
        with self.assertRaisesRegex(ValueError, "duplicate member"):
            read_samples(self.path)

    def test_reader_rejects_bad_index(self):
        def set_field(field, value):
            return lambda index: index.__setitem__(field, value)

        def set_sample_field(field, value):
            return lambda index: index["samples"][0].__setitem__(field, value)

        def set_call_field(field, value):
            return lambda index: index["samples"][0]["calls"][0].__setitem__(
                field, value
            )

        def set_intermediates(raw):
            return set_call_field("intermediates", raw)

        at = "samples[0].calls[0]"
        cases = [
            (lambda index: index.clear(), "unsupported version"),
            (set_field("version", "2.0"), "unsupported version"),
            (set_field("tol", [10**400, 0.0]), "too large"),
            (set_field("version", 1), "unsupported version"),
            (set_field("version", "1"), "unsupported version"),
            (set_field("version", "1.01"), "unsupported version"),
            (set_field("samples", []), "non-empty list"),
            (set_field("tol", [1e-4]), "tol must be"),
            (set_field("description", 1), "description must be a string"),
            (set_field("source", None), "source must be a string"),
            (set_sample_field("calls", []), "non-empty 'calls' list"),
            (set_sample_field("tol", [1e-4]), "tol must be"),
            (set_call_field("tol", [0.0, -1.0]), "tol must be"),
            (set_call_field("in", None), "'in' list"),
            (set_call_field("method", ""), "method must be a non-empty string"),
            (set_call_field("method", 1), "method must be a non-empty string"),
            (
                lambda index: index["samples"][0]["calls"][0].pop("method"),
                "method must be a non-empty string",
            ),
            (set_call_field("in", [f"{at}.in[0]", None]), "unsupported value"),
            (set_call_field("in", [f"{at}.in[0]", float("nan")]), "unsupported value"),
            (set_call_field("out", f"{at}.out[0]"), "out must be a list"),
            (set_intermediates([]), "intermediates must be an object"),
            (set_intermediates({"a.b": f"{at}.intermediates.a.b"}), "bad intermediate"),
            (set_intermediates({"linear": 1}), "bad intermediate"),
            (
                set_call_field("in", [f"{at}.in[7]"]),
                "must be 'samples\\[0\\].calls\\[0\\].in\\[0\\]'",
            ),
            (
                set_call_field("out", [f"{at}.out[0]", f"{at}.out[1]"]),
                "no tensor named",
            ),
            (set_call_field("out", [1]), "unreferenced tensors"),
        ]
        for edit, message in cases:
            with self.subTest(message=message):
                write_ptn(self.path, _PTG, {})
                write_samples(self.path, "golden", _golden())
                _edit_index(self.path, edit)
                with self.assertRaisesRegex(ValueError, message):
                    read_samples(self.path)

    def test_reader_names_set_on_unparseable_members(self):
        cases = [
            ({_INDEX: b'{"version": '}, "Expecting value"),
            ({_INDEX: b"[" * 100_000}, ""),
            (
                {
                    _INDEX: b'{"version": "1.0", "samples": [{"calls": [{"method": '
                    + b'"forward", "in": '
                    + b"[" * 600
                    + b"]" * 600
                    + b"}]}]}"
                },
                "recursion",
            ),
            ({_DATA: b"\x08" + bytes(7) + b"not json"}, ""),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                write_ptn(self.path, _PTG, {})
                write_samples(self.path, "golden", _golden())
                _rewrite(self.path, overrides)
                limit = sys.getrecursionlimit()
                sys.setrecursionlimit(1000)
                try:
                    with self.assertRaisesRegex(
                        ValueError, f"samples/golden.*{message}"
                    ):
                        read_samples(self.path)
                finally:
                    sys.setrecursionlimit(limit)

    def test_reader_accepts_newer_minor_version_and_unknown_fields(self):
        golden = _golden()
        write_samples(self.path, "golden", golden)

        def add_fields(index):
            index["version"] = "1.7"
            index["future"] = {"anything": 1}
            index["samples"][0]["future"] = True
            index["samples"][0]["calls"][0]["future"] = True

        _edit_index(self.path, add_fields)
        self.assertSetsEqual(read_samples(self.path)["golden"], golden)

    def test_newer_minor_may_reference_tensors_from_unknown_fields(self):
        write_samples(self.path, "golden", _golden())

        def move_outputs_to_unknown_field(index):
            index["version"] = "1.1"
            call = index["samples"][0]["calls"][0]
            call["future"] = call.pop("out")

        _edit_index(self.path, move_outputs_to_unknown_field)
        self.assertIsNone(read_samples(self.path)["golden"].samples[0].calls[0].outputs)
        _edit_index(self.path, lambda index: index.__setitem__("version", "1.0"))
        with self.assertRaisesRegex(ValueError, "unreferenced tensors"):
            read_samples(self.path)
