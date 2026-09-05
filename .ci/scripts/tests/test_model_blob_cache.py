# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the content-addressed model blob cache used by the CUDA workflows.
#
# Every case runs offline: S3 HEAD/GET and the aws CLI are mocked. The important
# invariants are that a blob is only removed from the artifact dir once it is
# known to be in S3, that stash never fails the producer job, and that restore
# either reproduces the exact bytes or fails.

import hashlib
import importlib.util
import io
import shutil
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def _load_cache():
    """Load the script by path, since .ci/scripts is not an importable package."""
    path = ROOT / ".ci" / "scripts" / "model_blob_cache.py"
    spec = importlib.util.spec_from_file_location("model_blob_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CACHE = _load_cache()
MODEL = "facebook/dinov2-small-imagenet1k-1-layer"
BLOB = b"weights" * 512
BLOB_SHA = hashlib.sha256(BLOB).hexdigest()
BLOB_KEY = (
    f"s3://gha-artifacts/pytorch/executorch/model-cache/{BLOB_SHA}/aoti_cuda_blob.ptd"
)
FRESH = (3, len(BLOB))
URL = "http://localhost/blob"
NOT_FOUND = urllib.error.HTTPError(URL, 404, "Not Found", {}, None)


class _Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


class StashTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.blob = self.root / "aoti_cuda_blob.ptd"
        self.blob.write_bytes(BLOB)
        self.pointer = self.root / "aoti_cuda_blob.ptd.sha256"
        self.model = self.root / "model.pte"
        self.model.write_bytes(BLOB)
        patches = [
            mock.patch.object(CACHE.shutil, "which", return_value="/usr/bin/aws"),
            mock.patch.object(CACHE.subprocess, "run"),
            mock.patch.object(CACHE, "cached_object", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        CACHE.subprocess.run.return_value = mock.Mock(returncode=0)

    def assert_blob_kept(self):
        self.assertEqual(self.blob.read_bytes(), BLOB)
        self.assertFalse(self.pointer.exists())

    def test_only_ptd_files_are_cached(self):
        CACHE.stash(self.root, MODEL)
        self.assertEqual(self.model.read_bytes(), BLOB)
        self.assertFalse((self.root / "model.pte.sha256").exists())
        self.assertTrue(self.pointer.exists())

    def test_cache_hit_writes_pointer_without_upload(self):
        CACHE.cached_object.return_value = FRESH
        CACHE.stash(self.root, MODEL)
        self.assertEqual(self.pointer.read_text(), f"{BLOB_SHA}  aoti_cuda_blob.ptd\n")
        self.assertFalse(self.blob.exists())
        CACHE.subprocess.run.assert_not_called()

    def test_miss_stale_or_wrong_size_uploads_then_writes_pointer(self):
        for cached in (None, (CACHE.REFRESH_AGE_DAYS + 5, len(BLOB)), (3, 1)):
            with self.subTest(cached=cached):
                self.blob.write_bytes(BLOB)
                self.pointer.unlink(missing_ok=True)
                CACHE.subprocess.run.reset_mock()
                CACHE.cached_object.return_value = cached
                CACHE.stash(self.root, MODEL)
                command = CACHE.subprocess.run.call_args.args[0]
                self.assertEqual(command[:3], ["aws", "s3", "cp"])
                self.assertEqual(command[-2:], [str(self.blob), BLOB_KEY])
                self.assertTrue(self.pointer.exists())
                self.assertFalse(self.blob.exists())

    def test_gated_model_keeps_blob_without_touching_s3(self):
        for model in ("google/gemma-3-4b-it", "unsloth/gemma-4-31B-it-GGUF"):
            with self.subTest(model=model):
                CACHE.stash(self.root, model)
                self.assert_blob_kept()
        CACHE.cached_object.assert_not_called()
        CACHE.subprocess.run.assert_not_called()

    def test_missing_aws_cli_keeps_blob(self):
        with mock.patch.object(CACHE.shutil, "which", return_value=None):
            CACHE.stash(self.root, MODEL)
        self.assert_blob_kept()
        CACHE.subprocess.run.assert_not_called()

    def test_failed_upload_keeps_blob(self):
        CACHE.subprocess.run.return_value = mock.Mock(returncode=1)
        CACHE.stash(self.root, MODEL)
        self.assert_blob_kept()

    def test_unsafe_name_keeps_blob(self):
        odd = self.root / "my blob.ptd"
        odd.write_bytes(BLOB)
        CACHE.stash(self.root, MODEL)
        self.assertEqual(odd.read_bytes(), BLOB)
        self.assertFalse((self.root / "my blob.ptd.sha256").exists())
        self.assertTrue(self.pointer.exists())

    def test_failure_on_one_blob_does_not_stop_the_others(self):
        other = self.root / "other.ptd"
        other.write_bytes(BLOB)
        real = CACHE.sha256_of

        def flaky(path):
            if path == self.blob:
                raise RuntimeError("boom")
            return real(path)

        with mock.patch.object(CACHE, "sha256_of", side_effect=flaky):
            CACHE.stash(self.root, MODEL)
        self.assert_blob_kept()
        self.assertFalse(other.exists())
        self.assertTrue((self.root / "other.ptd.sha256").exists())


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.pointer = self.root / "aoti_cuda_blob.ptd.sha256"
        self.pointer.write_text(f"{BLOB_SHA}  aoti_cuda_blob.ptd\n")
        self.blob = self.root / "aoti_cuda_blob.ptd"
        sleep = mock.patch.object(CACHE.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_round_trip(self):
        with mock.patch.object(
            CACHE.urllib.request, "urlopen", return_value=_Response(BLOB)
        ) as urlopen:
            CACHE.restore(self.root)
        self.assertEqual(self.blob.read_bytes(), BLOB)
        self.assertFalse(self.pointer.exists())
        self.assertEqual(
            urlopen.call_args.args[0],
            f"{CACHE.URL_BASE}/{BLOB_SHA}/aoti_cuda_blob.ptd",
        )

    def test_stash_output_restores(self):
        self.pointer.unlink()
        self.blob.write_bytes(BLOB)
        with mock.patch.object(CACHE, "cached_object", return_value=FRESH):
            CACHE.stash(self.root, MODEL)
        self.assertFalse(self.blob.exists())
        with mock.patch.object(
            CACHE.urllib.request, "urlopen", return_value=_Response(BLOB)
        ):
            CACHE.restore(self.root)
        self.assertEqual(self.blob.read_bytes(), BLOB)
        self.assertFalse(self.pointer.exists())

    def test_short_read_retries_then_fails_and_cleans_up(self):
        with mock.patch.object(
            CACHE.urllib.request,
            "urlopen",
            side_effect=lambda *a, **k: _Response(b"corrupt"),
        ) as urlopen:
            with self.assertRaises(SystemExit):
                CACHE.restore(self.root)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), [self.pointer.name]
        )

    def test_complete_body_with_wrong_digest_fails_without_retrying(self):
        with mock.patch.object(
            CACHE.urllib.request,
            "urlopen",
            side_effect=lambda *a, **k: _Response(b"corrupt", {"Content-Length": "7"}),
        ) as urlopen:
            with self.assertRaises(SystemExit):
                CACHE.restore(self.root)
        self.assertEqual(urlopen.call_count, 1)
        CACHE.time.sleep.assert_not_called()
        self.assertFalse(self.blob.exists())

    def test_missing_object_fails_without_retrying(self):
        with mock.patch.object(
            CACHE.urllib.request, "urlopen", side_effect=NOT_FOUND
        ) as urlopen:
            with self.assertRaises(SystemExit):
                CACHE.restore(self.root)
        self.assertEqual(urlopen.call_count, 1)
        CACHE.time.sleep.assert_not_called()

    def test_malformed_pointers_fail_without_downloading(self):
        for text in (
            f"{BLOB_SHA}  ..\n",
            f"{BLOB_SHA}  other.ptd\n",
            f"{BLOB_SHA}  aoti_cuda_blob.ptd extra\n",
            "nothex  aoti_cuda_blob.ptd\n",
            f"{BLOB_SHA}\n",
        ):
            with self.subTest(text=text):
                self.pointer.write_text(text)
                with mock.patch.object(CACHE.urllib.request, "urlopen") as urlopen:
                    with self.assertRaises(SystemExit):
                        CACHE.restore(self.root)
                urlopen.assert_not_called()

    def test_pointer_with_unsafe_name_is_rejected(self):
        self.pointer.unlink()
        odd = self.root / "a#b.ptd.sha256"
        odd.write_text(f"{BLOB_SHA}  a#b.ptd\n")
        with mock.patch.object(CACHE.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(SystemExit):
                CACHE.restore(self.root)
        urlopen.assert_not_called()

    def test_only_ptd_pointers_are_restored(self):
        stray = self.root / "checksums.sha256"
        stray.write_text("not a pointer\n")
        self.blob.write_bytes(BLOB)
        with mock.patch.object(CACHE.urllib.request, "urlopen") as urlopen:
            CACHE.restore(self.root)
        urlopen.assert_not_called()
        self.assertEqual(stray.read_text(), "not a pointer\n")
        self.assertFalse(self.pointer.exists())

    def test_existing_verified_blob_skips_download(self):
        self.blob.write_bytes(BLOB)
        with mock.patch.object(CACHE.urllib.request, "urlopen") as urlopen:
            CACHE.restore(self.root)
        urlopen.assert_not_called()
        self.assertFalse(self.pointer.exists())
        self.assertEqual(self.blob.read_bytes(), BLOB)

    def test_existing_stale_blob_is_replaced(self):
        self.blob.write_bytes(b"stale")
        with mock.patch.object(
            CACHE.urllib.request, "urlopen", return_value=_Response(BLOB)
        ):
            CACHE.restore(self.root)
        self.assertEqual(self.blob.read_bytes(), BLOB)

    def test_no_pointers_is_a_no_op(self):
        self.pointer.unlink()
        with mock.patch.object(CACHE.urllib.request, "urlopen") as urlopen:
            CACHE.restore(self.root)
        urlopen.assert_not_called()


class CachedObjectTest(unittest.TestCase):
    def test_age_and_size_from_headers(self):
        modified = datetime.now(timezone.utc) - timedelta(days=100, hours=1)
        headers = {"Last-Modified": format_datetime(modified), "Content-Length": "42"}
        with mock.patch.object(
            CACHE.urllib.request, "urlopen", return_value=_Response(b"", headers)
        ):
            self.assertEqual(CACHE.cached_object(URL), (100, 42))

    def test_missing_headers_count_as_absent(self):
        for headers in ({}, {"Content-Length": "42"}, {"Last-Modified": "garbage"}):
            with self.subTest(headers=headers):
                with mock.patch.object(
                    CACHE.urllib.request,
                    "urlopen",
                    return_value=_Response(b"", headers),
                ):
                    self.assertIsNone(CACHE.cached_object(URL))

    def test_missing_object_is_none(self):
        with mock.patch.object(CACHE.urllib.request, "urlopen", side_effect=NOT_FOUND):
            self.assertIsNone(CACHE.cached_object(URL))

    def test_blob_url_escapes_names(self):
        self.assertTrue(CACHE.blob_url("ab", "x y#z").endswith("/ab/x%20y%23z"))


if __name__ == "__main__":
    unittest.main()
