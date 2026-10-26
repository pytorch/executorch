"""Reuse a terminal, qualified native build for two matched mini exports.

Preserve the original build receipt and binary hashes. Path aliases in the
derived receipt resolve to those exact binaries, not rebuilt substitutes.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-job", "gguf", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    assert not args.output.exists()
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           candidate_accepted=False, full_model_export_started=False,
                                           **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    try:
        state("verifying-terminal-native-qualification")
        prior = json.loads((args.native_job / "progress.json").read_text())
        assert prior["status"] == "native-GPU-tests-passed"
        assert not (Path("/proc") / str(prior["controller_pid"])).exists()
        original_gate = args.native_job / "native-gate.json"
        gate = json.loads(original_gate.read_text())
        assert gate["native_GPU_tests_pass"] and gate["test_count"] >= 29
        assert len(gate["independent_session_tests"]) == 7
        spec = json.loads((package / "independent_session_capacity_source.json").read_text())
        assert gate["source_spec_SHA256"] == digest(package / "independent_session_capacity_source.json")
        source = Path(gate["source"])
        for name, record in spec["files"].items():
            assert digest(source / name) == record["after_SHA256"], name
        for name, expected in gate["unchanged_controls"].items():
            assert digest(source / name) == expected, name
        for name, expected in gate["binaries"].items():
            assert digest(Path(name)) == expected, name
        assert digest(args.native_job / "native-tests.xml") == gate["native_tests_xml_SHA256"]
        assert digest(args.gguf) == "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
        model_build = args.output / "model-build"
        model_build.symlink_to((args.native_job / "model-build").resolve(), target_is_directory=True)
        aliases = {}
        for name in ("muse_glimmer_runtime_kv_probe", "muse_glimmer_batching_worker"):
            old = args.native_job / "model-build" / name
            alias = model_build / name
            assert alias.resolve() == old.resolve()
            assert digest(alias) == gate["binaries"][str(old)]
            gate["binaries"][str(alias)] = digest(alias)
            aliases[str(alias)] = str(old)
        gate["imported_native_gate_SHA256"] = digest(original_gate)
        gate["verified_binary_path_aliases"] = aliases
        gate_path = args.output / "native-gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n")
        (args.output / "native-gate-import.json").write_text(json.dumps(dict(
            original_gate=str(original_gate), original_gate_SHA256=digest(original_gate),
            derived_gate_SHA256=digest(gate_path), binary_aliases=aliases,
            source_unchanged=True, native_binaries_unchanged=True), indent=2) + "\n")
        for role in ("mini-global", "mini-independent"):
            state("exporting-" + role, source=str(source), native_gate=str(gate_path))
            command = [sys.executable, "-B", "-u", str(package / "export_independent_session_mini.py"),
                       "--source", str(source), "--native-gate", str(gate_path), "--gguf", str(args.gguf),
                       "--role", role, "--output", str(args.output / (role + "-artifact")),
                       "--job-dir", str(args.output / (role + "-export")), "--lock", str(args.lock)]
            with (args.output / (role + "-controller.log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=21600)
        state("mini-artifacts-ready", source=str(source), native_gate=str(gate_path),
              automatic_reclamation_native_tested=True,
              source_spec_SHA256=gate["source_spec_SHA256"])
    except Exception as error:
        state("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
