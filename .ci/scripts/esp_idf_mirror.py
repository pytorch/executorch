# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Add a `compote component pack` archive to a local file:// component mirror.

Writes the layout `compote registry sync` produces:
  <mirror>/components/<namespace>/<name>.json
  <mirror>/components/<namespace>/<name>/<version>/<archive>, CHECKSUMS.json
so that a project with IDF_COMPONENT_STORAGE_URL=file://<mirror> resolves
<namespace>/<name> exactly as it would from the registry.
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from idf_component_tools.hash_tools.calculate import hash_dir
from idf_component_tools.hash_tools.checksums import ChecksumsManager
from idf_component_tools.manager import ManifestManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("mirror", type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    archive, mirror = args.archive.resolve(), args.mirror.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive) as tf:
            tf.extractall(tmp, filter="data")
        manifest = ManifestManager(tmp, args.name).load()
        component_hash = hash_dir(
            tmp,
            use_gitignore=manifest.use_gitignore,
            include=manifest.include_set,
            exclude=set(manifest.exclude_set)
            | {"**/.component_hash", "**/CHECKSUMS.json"},
            exclude_default=False,
        )
        version = str(manifest.version)
        rel_dir = Path("components") / args.namespace / args.name / version
        (mirror / rel_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, mirror / rel_dir / archive.name)
        ChecksumsManager(Path(tmp)).dump(mirror / rel_dir)

    dependencies = []
    for requirement in manifest.raw_requirements:
        # The registry evaluates rules and matches; this mirror does not.
        conditions = requirement.optional_requirement
        if conditions.matches or conditions.rules:
            raise SystemExit(
                f"{requirement.name} is a conditional dependency, which "
                f"{Path(__file__).name} cannot mirror"
            )
        namespace, _, name = requirement.name.rpartition("/")
        is_idf = requirement.source.type == "idf"
        dependencies.append(
            {
                "spec": requirement.version_spec,
                "source": requirement.source.type,
                "registry_url": requirement.registry_url,
                "is_public": requirement.is_public,
                "require": requirement.is_required,
                "name": None if is_idf else name,
                "namespace": None if is_idf else namespace,
                "rules": [],
                "matches": [],
            }
        )
    created_at = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": "local",
        "version": version,
        "created_at": created_at,
        "description": manifest.description,
        "component_hash": component_hash,
        "url": (rel_dir / archive.name).as_posix(),
        "targets": list(manifest.targets or []),
        "maintainers": [],
        "docs": {},
        "dependencies": dependencies,
        "license": None,
        "examples": [],
        "tags": [],
        "checksums": (rel_dir / "CHECKSUMS.json").as_posix(),
        "yanked_at": None,
        "yanked_message": None,
    }
    index = mirror / "components" / args.namespace / f"{args.name}.json"
    data = (
        json.loads(index.read_text())
        if index.exists()
        else {
            "name": args.name,
            "namespace": args.namespace,
            "created_at": created_at,
            "featured": False,
            "versions": [],
        }
    )
    data["versions"] = [v for v in data["versions"] if v["version"] != version]
    data["versions"].append(entry)
    index.write_text(json.dumps(data))
    print(f"{args.namespace}/{args.name} {version} -> {index}")


if __name__ == "__main__":
    main()
