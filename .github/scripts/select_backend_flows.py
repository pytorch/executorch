import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path


SHARED_PATHS = (
    ".ci/**",
    ".github/workflows/_test_backend.yml",
    ".github/scripts/select_backend_flows.py",
    ".github/scripts/test_select_backend_flows.py",
    "backends/test/**",
    "backends/transforms/**",
    "codegen/**",
    "exir/**",
    "runtime/**",
    "kernels/**",
    "extension/**",
    "schema/**",
    "third-party/**",
    "examples/models/**",
    "tools/cmake/**",
    "test/**",
    "CMakeLists.txt",
    "**/CMakeLists.txt",
    "CMakePresets.json",
    "setup.py",
    "pyproject.toml",
    "install_*",
    "requirements*",
    "torch_pin.py",
    "version.txt",
    ".gitmodules",
)


def select_flows(full, reduced, backend_paths, event, ref, changed, sampled):
    if (
        not isinstance(full, list)
        or not full
        or not all(isinstance(f, str) and f for f in full)
    ):
        raise ValueError("Full flows must be a nonempty list of flow names")
    if not isinstance(reduced, list) or not reduced or not set(reduced).issubset(full):
        raise ValueError("Reduced flows must be a nonempty subset of full flows")
    if event in ("schedule", "workflow_dispatch") or ref.startswith(
        ("refs/heads/release/", "refs/tags/")
    ):
        return full, "scheduled, release or forced run"
    if sampled:
        return full, "existing full-run sample"
    if event not in ("pull_request", "push") or (
        event == "push" and ref != "refs/heads/main"
    ):
        return full, "unrecognized event or branch"
    if changed is None:
        return full, "changed files unavailable"
    if (
        not isinstance(backend_paths, list)
        or not backend_paths
        or not all(isinstance(p, str) and p for p in backend_paths)
    ):
        return full, "backend paths unavailable"
    for path in changed:
        if any(
            fnmatch.fnmatchcase(path, pattern)
            for pattern in (*SHARED_PATHS, *backend_paths)
        ):
            return full, f"relevant change: {path}"
    return reduced, "unrelated changes"


def changed_files(event, payload, source_sha):
    if event == "pull_request":
        pr = payload.get("pull_request", {})
        base = pr.get("base", {}).get("sha", "")
        head = pr.get("head", {}).get("sha", "")
        revision = f"{base}...{head}"
    elif event == "push":
        base = payload.get("before", "")
        head = payload.get("after", "")
        revision = f"{base}..{head}"
    else:
        return None
    if head != source_sha or not all(
        re.fullmatch(r"[0-9a-f]{40}", sha) and sha != "0" * 40 for sha in (base, head)
    ):
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", revision, "--"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None
    return [path.decode() for path in result.stdout.split(b"\0") if path]


def main():
    event = os.environ["GITHUB_EVENT_NAME"]
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    flows, reason = select_flows(
        json.loads(os.environ["FULL_FLOWS"]),
        json.loads(os.environ["REDUCED_FLOWS"]),
        json.loads(os.environ["BACKEND_PATHS"]),
        event,
        os.environ["GITHUB_REF"],
        changed_files(event, payload, os.environ["SOURCE_SHA"]),
        os.environ["IS_FULL_RUN"] == "true",
    )
    print(reason)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("flows=" + json.dumps(flows, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
