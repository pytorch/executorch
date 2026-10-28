"""Build a graph-disabled transport and repeat the paired mini boundary diagnosis."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(config, output):
    cfg = json.loads(config.read_text())
    output.mkdir(exist_ok=False)
    (output / "configuration.json").write_text(json.dumps(cfg, indent=2))
    for name, expected in cfg["source_SHA256"].items():
        assert digest(cfg[name]) == expected, name
    source = Path(cfg["probe_source"]).read_text()
    original = "config.options.cuda_graph_for_static_methods = true;"
    assert source.count(original) == 1
    source = source.replace(original, "config.options.cuda_graph_for_static_methods = false;")
    probe_source = output / "mini_probe.cpp"
    probe_source.write_text(source)
    flags = {}
    for line in Path(cfg["compile_flags"]).read_text().splitlines():
        if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
            name, value = line.split("=", 1)
            flags[name.strip()] = shlex.split(value)
    assert set(flags) == {"CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"}
    obj = output / "mini_probe.o"
    compile_command = ["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                       *flags["CXX_FLAGS"], "-c", str(probe_source), "-o", str(obj)]
    build = Path(cfg["probe_build"])
    link_command = shlex.split(Path(cfg["link_command"]).read_text())
    old_objects = [arg for arg in link_command if arg.endswith("mini_probe.cpp.o")]
    assert len(old_objects) == 1
    old_backend = cfg["old_backend_archive"]
    assert link_command.count(old_backend) == cfg["backend_archive_occurrences"]
    link_command = [str(obj) if arg == old_objects[0] else
                    cfg["backend_archive"] if arg == old_backend else arg
                    for arg in link_command if not arg.startswith("-Wl,--dependency-file=")]
    probe = output / "graph-disabled-probe"
    link_command[link_command.index("-o") + 1] = str(probe)
    link_command.append("-lcuda")
    commands = [compile_command, link_command]
    (output / "build-commands.json").write_text(json.dumps(commands, indent=2))
    with (output / "build.log").open("x") as log:
        for command in commands:
            subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "build-receipt.json").write_text(json.dumps({
        "probe_SHA256": digest(probe), "source_SHA256": digest(probe_source),
        "backend_archive_SHA256": digest(cfg["backend_archive"]),
        "only_probe_source_change": "cuda_graph_for_static_methods=false",
        "native_runtime_unchanged": True, "PTE_and_PTD_unchanged": True,
        "diagnostic_only": True,
    }, indent=2))
    matrix = dict(cfg["matrix_configuration"])
    matrix["probe"] = str(probe)
    matrix["expected_SHA256"] = dict(matrix["expected_SHA256"], probe=digest(probe))
    matrix_path = output / "matrix-configuration.json"
    matrix_path.write_text(json.dumps(matrix, indent=2))
    subprocess.run([cfg["python"], "-B", "-u", cfg["boundary_script"],
                    "--config", str(matrix_path), "--output", str(output / "matrix")], check=True)
    result = json.loads((output / "matrix/result.json").read_text())
    result.update(graphs_disabled=True, diagnostic_only=True,
                  full_model_qualified=False, original_qualification_suite_passed=False)
    (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rc = 1
    try:
        main(args.config, args.output)
        rc = 0
    except BaseException:
        if args.output.exists():
            (args.output / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if args.output.exists():
            (args.output / "exit-receipt.json").write_text(json.dumps({
                "returncode": rc, "diagnostic_only": True,
                "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }))
    raise SystemExit(rc)
