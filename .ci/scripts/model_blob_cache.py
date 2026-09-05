#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Content-addressed S3 cache for exported model weight blobs.

CI re-exports the same models on every run, and the multi-GB weight blobs
(aoti_cuda_blob.ptd) come out byte-identical run after run while only the
small model.pte changes. Shipping those blobs as GitHub artifacts each time
is what dominates the repo's artifact storage.

    stash --model <org/name> <dir>
                   Replace every *.ptd under <dir> with a <name>.sha256 pointer
                   (sha256sum format) once the blob is stored at
                   s3://gha-artifacts/pytorch/executorch/model-cache/<sha256>/<name>.
                   Blobs already there are not re-uploaded.
    restore <dir>  Download the blob behind every *.ptd.sha256 under <dir>,
                   verify the digest and delete the pointer.

Uploads need the aws CLI and write credentials (the OIDC role on OSDC runners
or the EC2 instance profile); reads are anonymous. Anything that prevents an
upload leaves the file in place with a warning, so the job degrades to a plain
artifact instead of failing. Only *.ptd files are ever uploaded, so nothing
else that lands in the artifact directory can end up in the public bucket, and
blobs exported from gated models stay in the artifact altogether.

The bucket is writable by CI jobs across the org, so a cached object can be
overwritten by anyone with that access. Consumers verify the digest from the
pointer before a blob is used, so a corrupted object can only fail the job,
never feed it other bytes.
"""

import argparse
import hashlib
import http.client
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

BUCKET = "gha-artifacts"
PREFIX = "pytorch/executorch/model-cache"
URL_BASE = f"https://{BUCKET}.s3.us-east-1.amazonaws.com/{PREFIX}"
BLOB_SUFFIX = ".ptd"
POINTER_SUFFIX = ".sha256"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
# The pointer format is "<digest>  <name>", so names must be free of whitespace.
NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
# Gemma weights are gated on Hugging Face and must not become anonymously
# downloadable, so anything derived from them stays in the GitHub artifact.
GATED_MODEL_RE = re.compile(r"gemma", re.IGNORECASE)
# The bucket lifecycle deletes objects under pytorch/ 90 days after they were
# written, regardless of use. Re-uploading anything older than this keeps hot
# blobs alive and leaves every run's pointers valid for at least 30 days.
REFRESH_AGE_DAYS = 60
UPLOAD_TIMEOUT_S = 3600
CHUNK = 8 << 20


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blob_url(digest, name):
    return f"{URL_BASE}/{digest}/{urllib.parse.quote(name, safe='')}"


def cached_object(url):
    """(age in days, size in bytes) of the object at url, or None if absent."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            modified = parsedate_to_datetime(resp.headers.get("Last-Modified"))
            size = int(resp.headers.get("Content-Length"))
            return (datetime.now(timezone.utc) - modified).days, size
    except (OSError, http.client.HTTPException, TypeError, ValueError):
        return None


def stash_one(path):
    if not NAME_RE.fullmatch(path.name):
        raise RuntimeError("file name cannot be stored in a pointer")
    digest = sha256_of(path)
    size = path.stat().st_size
    key = f"{digest}/{path.name}"
    cached = cached_object(blob_url(digest, path.name))
    print(f"{path}: sha256={digest} size={size} cached={cached}", flush=True)
    # A size mismatch means the key holds something other than this blob;
    # re-uploading overwrites it rather than trusting it.
    if cached is None or cached[0] >= REFRESH_AGE_DAYS or cached[1] != size:
        if shutil.which("aws") is None:
            raise RuntimeError("aws CLI not found")
        upload = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                "--only-show-errors",
                "--region",
                "us-east-1",
                str(path),
                f"s3://{BUCKET}/{PREFIX}/{key}",
            ],
            timeout=UPLOAD_TIMEOUT_S,
        )
        if upload.returncode != 0:
            raise RuntimeError(f"aws s3 cp exited {upload.returncode}")
        print(f"uploaded {key}", flush=True)
    else:
        print(f"cache hit, skipped upload of {key}", flush=True)
    path.with_name(path.name + POINTER_SUFFIX).write_text(f"{digest}  {path.name}\n")
    path.unlink()


def stash(root, model):
    if GATED_MODEL_RE.search(model):
        print(f"::notice::model blob cache: skipped, {model} is gated", flush=True)
        return
    stashed = kept = 0
    for path in sorted(p for p in root.rglob("*" + BLOB_SUFFIX) if p.is_file()):
        try:
            stash_one(path)
            stashed += 1
        except Exception as e:  # never fail the producer job over the cache
            kept += 1
            print(f"::warning::keeping {path.name} in the artifact: {e!r}", flush=True)
            traceback.print_exc(file=sys.stdout)
    print(
        f"::notice::model blob cache: {stashed} blob(s) stashed, {kept} kept in the artifact",
        flush=True,
    )


def download(url, dest, digest):
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(3):
        if attempt:
            time.sleep(30 * attempt)
        retry = True
        try:
            actual = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(url, timeout=60) as resp, partial.open(
                "wb"
            ) as out:
                length = resp.headers.get("Content-Length")
                for chunk in iter(lambda: resp.read(CHUNK), b""):
                    actual.update(chunk)
                    out.write(chunk)
                    received += len(chunk)
            if actual.hexdigest() == digest:
                partial.replace(dest)
                return
            error = f"sha256 mismatch after {received} bytes, got {actual.hexdigest()}"
            # Retry short reads only; a complete body is what the object really holds.
            retry = length is None or received != int(length)
        except urllib.error.HTTPError as e:
            error = str(e)
            retry = e.code not in (403, 404)
        except (OSError, http.client.HTTPException) as e:
            error = str(e)
        print(f"download of {url} failed: {error}", flush=True)
        if not retry:
            break
    partial.unlink(missing_ok=True)
    print(f"::error::could not restore {dest} from {url}", flush=True)
    sys.exit(1)


def restore(root):
    for pointer in sorted(root.rglob("*" + BLOB_SUFFIX + POINTER_SUFFIX)):
        fields = pointer.read_text().split()
        stem = pointer.name[: -len(POINTER_SUFFIX)]
        if (
            len(fields) != 2
            or not DIGEST_RE.fullmatch(fields[0])
            or fields[1] != stem
            or not NAME_RE.fullmatch(stem)
        ):
            print(f"::error::malformed pointer {pointer}: {fields!r}", flush=True)
            sys.exit(1)
        digest, name = fields
        dest = pointer.with_name(name)
        if dest.exists() and sha256_of(dest) == digest:
            pointer.unlink()
            continue
        url = blob_url(digest, name)
        print(f"restoring {dest} from {url}", flush=True)
        download(url, dest, digest)
        pointer.unlink()
        print(f"restored {dest} ({dest.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["stash", "restore"])
    parser.add_argument("dir", type=Path)
    parser.add_argument("--model", help="Hugging Face id of the exported model")
    args = parser.parse_args()
    if not args.dir.is_dir():
        parser.error(f"{args.dir} is not a directory")
    if args.command == "restore":
        restore(args.dir)
    elif args.model is None:
        parser.error("stash requires --model")
    else:
        stash(args.dir, args.model)


if __name__ == "__main__":
    main()
