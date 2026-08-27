import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import pandas as pd
import torch
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchProfileLevel,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.check_qnn_version import (
    is_qnn_sdk_version_less_than,
)
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte

from graphviz import Digraph


class DrawGraph:
    def __init__(
        self,
        filename: str,
        directory: str,
        py_op_wrapper_list: [PyQnnManager.PyQnnOpWrapper],
        dot_string=False,
    ):
        self.filename = filename
        self.directory = directory
        self.py_op_wrapper_list = py_op_wrapper_list
        self.dot = Digraph(filename, format="svg")
        self.dot.attr(rankdir="TB")
        self.dot_string = dot_string
        self.draw()

    def dfs_add_edges(self, node_name, visited, node_list):
        if node_name in visited:
            return
        visited.add(node_name)

        input_list = node_list[node_name]["input_list"]
        for input_node_name in input_list:
            self.dot.edge(input_node_name, node_name)
            self.dfs_add_edges(input_node_name, visited, node_list)

    def get_dot_graph(self):
        visited = set()
        node_list = {}
        excel_data = []

        self.get_node(node_list)
        self.add_node(node_list, excel_data)
        self.to_excel(excel_data)

        # add edge
        for node_name, _ in node_list.items():
            if node_name not in visited:
                self.dfs_add_edges(node_name, visited, node_list)

        return self.dot

    def get_node(self, node_list):
        for py_op_wrapper in self.py_op_wrapper_list:
            op_wrapper = py_op_wrapper.GetOpWrapper()
            # TODO: multi output
            for i in range(op_wrapper.GetOpConfig()["numOfOutputs"]):
                if op_wrapper.GetOpConfig()["outputTensors"][0].version == 2:
                    node = op_wrapper.GetOpConfig()["outputTensors"][i].v2
                    node_name = node.name
                    input_list = []
                    for j in range(op_wrapper.GetOpConfig()["numOfInputs"]):
                        if op_wrapper.GetOpConfig()["inputTensors"][j].version == 2:
                            input_node = op_wrapper.GetOpConfig()["inputTensors"][j].v2
                            input_node_name = input_node.name
                            if input_node_name not in node_list:
                                node_list[input_node_name] = {
                                    "node": input_node,
                                    "input_list": [],
                                }
                            input_list.append(input_node_name)
                        else:
                            raise ValueError("Unsupported tensor version")
                    if node_name not in node_list:
                        node_list[node_name] = {"node": node, "input_list": input_list}
                    else:
                        node_list[node_name]["input_list"] = input_list
                else:
                    raise ValueError("Unsupported tensor version")

    def add_node(self, node_list, excel_data):
        for node_name, tensor in node_list.items():
            node = tensor["node"]
            name = node_name
            data_type = node.dataType
            tensor_type = node.type
            dims = node.dimensions
            quantization_encoding = node.quantizeParams.quantizationEncoding
            scale = []
            offset = []
            if (
                quantization_encoding
                == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            ):
                scale.append(node.quantizeParams.scaleOffsetEncoding.scale)
                offset.append(node.quantizeParams.scaleOffsetEncoding.offset)
            elif (
                quantization_encoding
                == PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
            ):
                for i in range(
                    node.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets
                ):
                    scale.append(
                        node.quantizeParams.axisScaleOffsetEncoding.scaleOffset[i].scale
                    )
                    offset.append(
                        node.quantizeParams.axisScaleOffsetEncoding.scaleOffset[
                            i
                        ].offset
                    )
            excel_data.append(
                {
                    "name": name,
                    "tensor_type": tensor_type,
                    "scale": scale,
                    "offset": offset,
                }
            )
            # Default color for intermediate nodes
            bg_color = "white"
            if "input" in node_name or "output" in node_name:
                bg_color = "lightgreen"
            elif tensor_type == 4:
                bg_color = "lightpink"
            label = f"""<
                <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR><TD BGCOLOR="{bg_color}">name: {name}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">data_type: {data_type}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">tensor_type: {tensor_type}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">dims: {dims}</TD></TR>
                <TR><TD BGCOLOR="{bg_color}">quantization_encoding: {quantization_encoding}</TD></TR>
            """
            label += "</TABLE>>"
            self.dot.node(
                node_name,
                label,
                shape="box",
                style="rounded",
                fillcolor="transparent",
                color="black",
            )

    def to_excel(self, excel_data):
        param_rows = []
        activation_rows = []

        for entry in excel_data:
            name = entry["name"]
            scale = entry["scale"]
            offset = entry["offset"]
            if (
                entry["tensor_type"]
                == PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC
            ):
                param_rows.append({"name": name, "scale": scale, "offset": offset})
            else:
                activation_rows.append({"name": name, "scale": scale, "offset": offset})
        param_df = pd.DataFrame(param_rows)
        scale_df = pd.DataFrame(activation_rows)
        output_file = f"{self.filename}.xlsx"

        with pd.ExcelWriter(output_file) as writer:
            param_df.to_excel(writer, sheet_name="Parameters", index=False)
            scale_df.to_excel(writer, sheet_name="Scales", index=False)

    def draw(self):
        graph = self.get_dot_graph()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_directory = f"{tmp_dir}/outputs"
            graph.render(
                self.filename,
                directory=temp_directory,
                format="svg",
                cleanup=not self.dot_string,
            )
            source_file = os.path.join(temp_directory, f"{self.filename}.svg")
            destination_file = os.path.join(self.directory, f"{self.filename}.svg")
            shutil.move(source_file, destination_file)
            if self.dot_string:
                dot_file = os.path.join(temp_directory, f"{self.filename}")
                dot_dest_file = os.path.join(self.directory, f"{self.filename}.dot")
                shutil.move(dot_file, dot_dest_file)


@dataclass(frozen=True)
class QnnHtpProfileArtifacts:
    """Files emitted for one HTP optrace/hextimate run.

    Data fields:
    - binary_path: dumped `.dlc` or `.bin` input.
    - mode: `"optrace"` for on-device hardware counters, or `"hextimate"`
      for host-only compile-time estimation.
    - prepare_mode: `"online"` for `.dlc` from `QnnConfig.online_prepare=True`,
      or `"offline"` for `.bin` from `online_prepare=False`.
    - chrometrace_json: Chrome trace JSON, viewable with `chrome://tracing` or
      Perfetto.
    - qhas_json: QHAS JSON
    - qhas_html: QHAS HTML report.
    - htp_graph_json: HTP graph JSON after optimization.
    - htp_graph_before_json: HTP graph JSON before optimization.
    - runtrace_json: runtrace JSON for optrace, or None when not emitted.

    Use visualizer_reports() for the subset safe to pass to
    qairt_visualizer.view(reports=...).
    """

    binary_path: str
    mode: Literal["optrace", "hextimate"]
    prepare_mode: Literal["online", "offline"]
    chrometrace_json: str
    qhas_json: Optional[str]
    qhas_html: str
    htp_graph_json: str
    htp_graph_before_json: str
    runtrace_json: Optional[str]

    def visualizer_reports(self) -> List[str]:
        """Reports safe to pass to qairt_visualizer.view(reports=...)."""
        reports = [self.chrometrace_json]
        if self.qhas_json is not None:
            reports.append(self.qhas_json)
        return reports


#   Hextimate (compile-time perf estimation) requires SDK >= 2.41. Below that,
#   qnn-context-binary-generator silently drops the hextimate parameters
_MIN_SDK_FOR_HEXTIMATE = "2.41"
_HEXTIMATE_SUPPORTED_SOCS = (
    QcomChipset.SA8540,
    QcomChipset.SA8255,
    QcomChipset.QCS9100,
    QcomChipset.SA8797,
)


#   - QNN: libQnnHtpNetRunExtensions.so
#   - QAIRT: libQairtHtpBackendExtensions.so (QAIRT 2.49+ sdk)
_BACKEND_EXTENSIONS_LIB = "libQnnHtpNetRunExtensions.so"


class QnnTool:
    """Host-side wrapper around the QNN profiling CLI toolchain.

    Compatibility (see README.md §QAIRT Profiling for the user-facing table):
        - Optrace: SDK 2.37+
        - Hextimate: SDK 2.41+ (gate and raise error)

    """

    def __init__(
        self,
        artifact_dir,
        soc_id,
        adb,
        sample_input=None,
        build_folder=None,
        workspace="/data/local/tmp/qnn_executorch_test",
    ):
        self.qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
        self.ndk = os.environ.get("ANDROID_NDK_ROOT", None)
        assert self.qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        assert self.ndk, "ANDROID_NDK_ROOT was not found in environment variable"

        self.artifact_dir = artifact_dir
        self.workspace = workspace
        self.adb = adb
        self.sample_input = sample_input
        self.build_folder = build_folder
        self.soc_id = soc_id

    def _get_base_config(self):
        # Generate base device profile — every subprocess call clones from this.
        return {
            "backend_extensions": {
                "config_file_path": os.path.join(self.artifact_dir, "config.json"),
            },
            "config": {
                "devices": [
                    {
                        "profiling_level": "linting",
                        "cores": [
                            {"perf_profile": "burst", "rpc_control_latency": 100}
                        ],
                        "soc_id": int(self.soc_id),
                    }
                ]
            },
        }

    def _write_config_files(
        self, backend_extensions: Optional[dict] = None
    ) -> Tuple[str, str]:
        """Write backend_extension_config.json and config.json in artifact_dir.

        Returns (backend_ext_path, config_path).
        """

        if backend_extensions is None:
            backend_extensions = self._get_base_config()["backend_extensions"]

        backend_ext_path = os.path.join(
            self.artifact_dir, "backend_extension_config.json"
        )
        config_path = os.path.join(self.artifact_dir, "config.json")
        with open(backend_ext_path, "w") as f:
            json.dump({"backend_extensions": backend_extensions}, f, indent=4)
        with open(config_path, "w") as f:
            json.dump(self._get_base_config()["config"], f, indent=4)
        return backend_ext_path, config_path

    def _run(self, cmd: List[str], step: str) -> None:
        """Run a subprocess with argv list; assert on non-zero exit."""
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"{step} failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    def _qnn_context_binary_generator(
        self,
        qnn_binary_file: str,
        binary_name: str,
        enable_hextimate: bool,
    ) -> None:
        target = "x86_64-linux-clang"
        backend_ext = self._get_base_config()["backend_extensions"]
        if enable_hextimate:
            backend_ext["shared_library_path"] = (
                f"{self.qnn_sdk}/lib/{target}/{_BACKEND_EXTENSIONS_LIB}"
            )
        backend_ext_path, _ = self._write_config_files(backend_ext)

        cmd = [
            f"{self.qnn_sdk}/bin/{target}/qnn-context-binary-generator",
            "--backend",
            f"{self.qnn_sdk}/lib/{target}/libQnnHtp.so",
            "--model",
            f"{self.qnn_sdk}/lib/{target}/libQnnModelDlc.so",
            "--dlc_path",
            os.path.join(self.artifact_dir, qnn_binary_file),
            "--config_file",
            backend_ext_path,
            "--binary_file",
            binary_name,
            "--output_dir",
            self.artifact_dir,
            "--profiling_level",
            "detailed",
            "--profiling_option",
            "optrace",
        ]
        self._run(cmd, "qnn-context-binary-generator")
        expected = os.path.join(self.artifact_dir, f"{binary_name}.bin")
        assert os.path.isfile(
            expected
        ), f"qnn-context-binary-generator ran but did not produce {expected}"

    def _qnn_net_run(self, graph_name: str) -> None:
        # backend-extensions library path is device-relative when running via adb
        backend_ext = {
            "shared_library_path": f"./{_BACKEND_EXTENSIONS_LIB}",
            "config_file_path": "config.json",
        }
        backend_ext_path, config_path = self._write_config_files(backend_ext)

        target = "aarch64-android"
        files = [
            f"{self.qnn_sdk}/lib/{target}/{_BACKEND_EXTENSIONS_LIB}",
            backend_ext_path,
            config_path,
            os.path.join(self.artifact_dir, f"{graph_name}.bin"),
            f"{self.qnn_sdk}/bin/{target}/qnn-net-run",
        ]
        cmds = [
            f"export LD_LIBRARY_PATH={self.workspace} &&",
            f"export ADSP_LIBRARY_PATH={self.workspace} &&",
            f"cd {self.workspace} &&",
            "./qnn-net-run",
            "--backend libQnnHtp.so",
            "--input_list input_list.txt",
            f"--retrieve_context {graph_name}.bin",
            "--use_native_input_files",
            "--use_native_output_files",
            "--config_file backend_extension_config.json",
            "--profiling_level detailed",
            "--profiling_option optrace",
        ]
        self.adb.push(
            inputs=self.sample_input,
            files=files,
        )
        self.adb.execute(custom_runner_cmd=" ".join(cmds))
        self.adb._adb(
            [
                "pull",
                "-a",
                f"{self.workspace}/output/qnn-profiling-data_0.log",
                self.artifact_dir,
            ]
        )

        assert os.path.isfile(
            f"{self.artifact_dir}/qnn-profiling-data_0.log"
        ), f"Error: qnn-profiling-data_0.log not found in {self.artifact_dir}"

    def _qnn_profile_viewer(self, schematic_stem: str, graph_idx: int) -> None:
        # profile-viewer takes its own config schema (`features`), not the
        # device profile schema. Written to the SAME file name because that's
        # the flag qnn-profile-viewer expects — a per-step fresh config avoids
        # any leak from earlier CBG/net-run configs.
        backend_ext_path = os.path.join(
            self.artifact_dir, "backend_extension_config.json"
        )
        with open(backend_ext_path, "w") as f:
            json.dump({"features": {"qhas_json": True}}, f, indent=4)

        target = "x86_64-linux-clang"
        # TODO: remove assumption that AOT dumpped schematic file exists in same cwd
        # we need to make .pte self-contained.
        schematic = os.path.join(os.getcwd(), f"{schematic_stem}.bin")
        assert os.path.isfile(schematic), (
            f"qnn-profile-viewer expected schematic at {schematic}; "
            "in case of online_prepare, the context-binary-generator step should have produced it in artifact_dir. "
            "in case of offline_prepare, the schematic should be dumpped from pte, make sure profiling_level=3 when generating pte. "
        )

        cmd = [
            f"{self.qnn_sdk}/bin/{target}/qnn-profile-viewer",
            "--config",
            backend_ext_path,
            "--schematic",
            schematic,
            "--reader",
            f"{self.qnn_sdk}/lib/{target}/libQnnHtpOptraceProfilingReader.so",
            "--input_log",
            os.path.join(self.artifact_dir, "qnn-profiling-data_0.log"),
            "--output",
            os.path.join(self.artifact_dir, f"optrace_{graph_idx}.json"),
        ]
        self._run(cmd, "qnn-profile-viewer")

    def _validated_qhas_json(self, qhas_path: str) -> Optional[str]:
        """Return path if the QHAS JSON parses; None if truncated (hextimate SDK bug).

        Note: QHAS JSON is truncated for hextimate mode (SDK bug, still open as of
        2.50 nightly). When time_us == 0, the SDK divides 1e6 / 0 = +Infinity,
        rapidjson rejects the write, and the JSON stream is cut at ~3900 bytes
        with "inf_per_s": <no value>. We detect this and return qhas_json=None
        rather than repair — the HTML report and chrometrace remain valid.
        """
        if not os.path.isfile(qhas_path):
            return None
        try:
            with open(qhas_path, "r") as f:
                json.load(f)
            return qhas_path
        except json.JSONDecodeError:
            return None

    def run(
        self,
        mode: Literal["optrace", "hextimate"],
        binary_file: str,
    ) -> QnnHtpProfileArtifacts:
        """Run qnn-profile-viewer for one dumped .dlc/.bin.

        Docs:
        - Optrace:   https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html#qnn-htp-optrace-profiling
        - Hextimate: https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-hextimate-profiling
        - QHAS:      https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html#qnn-htp-analysis-summary-qhas
        """
        assert mode in ("optrace", "hextimate"), f"unknown mode {mode!r}"

        graph_name, ext = os.path.splitext(binary_file)
        if mode == "optrace":
            assert ext in (".dlc", ".bin"), (
                f"optrace supports .dlc (online prepare) and .bin (offline prepare); "
                f"got {ext!r}"
            )
        else:  # hextimate
            assert ext == ".dlc", (
                f"hextimate requires .dlc (online prepare); got {ext!r}. "
                "For offline-prepare context binaries, use mode='optrace' instead."
            )
            if is_qnn_sdk_version_less_than(_MIN_SDK_FOR_HEXTIMATE):
                # SDK < 2.41 silently drops hextimate config and produces a
                # standard context binary — fail loudly before that trap.
                raise AssertionError(
                    f"hextimate requires QNN SDK >= {_MIN_SDK_FOR_HEXTIMATE}; "
                    f"the current SDK at $QNN_SDK_ROOT={self.qnn_sdk} is older. "
                    "Older SDKs silently ignore hextimate parameters and emit "
                    "an ordinary profiling log with no hextimate events."
                )

        prepare_mode: Literal["online", "offline"] = (
            "online" if ext == ".dlc" else "offline"
        )

        # Extract graph index if the file follows the "<name>_<n>" convention.
        match = re.match(r"^(.*)_(\d+)$", graph_name)
        if match:
            graph_base_name = match.group(1)
            graph_idx = int(match.group(2))
        else:
            graph_base_name = graph_name
            graph_idx = 0

        # Step 1: for online-prepare (.dlc), materialize the context binary +
        # schematic on the host. Offline-prepare (.bin) already has both.
        if ext == ".dlc":
            self._qnn_context_binary_generator(
                qnn_binary_file=binary_file,
                binary_name=f"{graph_base_name}.serialized",
                enable_hextimate=(mode == "hextimate"),
            )
            graph_name = f"{graph_base_name}.serialized"

        # Step 2: for optrace, run on device. Hextimate is host-only.
        if mode == "optrace":
            self._qnn_net_run(graph_name=graph_name)

        # Step 3: post-process into optrace.json + QHAS side-artifacts.
        self._qnn_profile_viewer(
            schematic_stem=f"{graph_base_name}_schematic",
            graph_idx=graph_idx,
        )

        # Collect the six output files. qnn-profile-viewer names them by
        # stripping `.json` off --output and appending suffixes.
        base = os.path.join(self.artifact_dir, f"optrace_{graph_idx}")
        chrometrace_json = f"{base}.json"
        qhas_json_candidate = f"{base}_qnn_htp_analysis_summary.json"
        qhas_html = f"{base}_qnn_htp_analysis_summary.html"
        htp_graph_json = f"{base}_htp.json"
        htp_graph_before_json = f"{base}_htp_graph_before.json"
        runtrace_json = f"{base}_runtrace.json"

        assert os.path.isfile(
            chrometrace_json
        ), f"qnn-profile-viewer did not produce {chrometrace_json}"

        return QnnHtpProfileArtifacts(
            binary_path=os.path.join(self.artifact_dir, binary_file),
            mode=mode,
            prepare_mode=prepare_mode,
            chrometrace_json=chrometrace_json,
            qhas_json=self._validated_qhas_json(qhas_json_candidate),
            qhas_html=qhas_html,
            htp_graph_json=htp_graph_json,
            htp_graph_before_json=htp_graph_before_json,
            runtrace_json=(runtrace_json if os.path.isfile(runtrace_json) else None),
        )


def _validate_pte_profile_level(pte_path: str) -> None:
    """Assert that any offline-prepare .pte was built with profile_level=3."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(pte_path, "rb") as f:
        program = deserialize_pte_binary(f.read()).program

    for execution_plan in program.execution_plan:
        for delegate in execution_plan.delegates:
            if delegate.id != "QnnBackend":
                continue
            spec = delegate.compile_specs[0]
            options = flatbuffer_to_option(bytes(spec.value))
            if options.online_prepare:
                continue  # online-prepare: profile_level is set on-host later
            assert options.profile_level == QnnExecuTorchProfileLevel.kProfileOptrace, (
                f"{pte_path} was compiled with online_prepare=False and "
                f"profile_level={options.profile_level.name}."
                "HTP Profling (Optrace) feature requires profile_level=3 (kProfileOptrace) "
                "at build_executorch_binary() time — \n"
                "Please re-export with qnn_config.profile_level=3, or use online_prepare=True"
            )


def _validate_hextimate_soc(soc_id: QcomChipset) -> None:
    if soc_id not in _HEXTIMATE_SUPPORTED_SOCS:
        supported = ", ".join(soc.name for soc in _HEXTIMATE_SUPPORTED_SOCS)
        raise AssertionError(
            f"hextimate currently supports only {supported}; got {soc_id.name}."
        )


def _generate_htp_analysis_result(
    artifact_dir: str,
    soc_id: QcomChipset,
    pte_path: str,
    mode: Literal["optrace", "hextimate"],
    inputs: Optional[Sequence[Tuple[torch.Tensor]]] = None,
    adb=None,
) -> List[QnnHtpProfileArtifacts]:
    assert mode in ("optrace", "hextimate"), f"unknown mode {mode!r}"
    if mode == "optrace":
        assert adb is not None, "optrace requires adb for on-device execution"
    _validate_pte_profile_level(pte_path)

    dumpfiles = dump_context_from_pte(pte_path, output_dir=artifact_dir)

    qnn_tool = QnnTool(
        artifact_dir=artifact_dir,
        sample_input=inputs,
        soc_id=soc_id,
        adb=adb,
        build_folder=(adb.build_path if adb is not None else None),
        workspace=(adb.workspace if adb is not None else None),
    )

    return [qnn_tool.run(mode=mode, binary_file=os.path.basename(f)) for f in dumpfiles]


# backward compatibility shim
def generate_optrace(
    artifact,
    soc_id: QcomChipset,
    adb,
    pte_path: str,
    inputs: Sequence[Tuple[torch.Tensor]],
):
    """see generate_htp_profile_result()"""
    return generate_htp_profile_result(artifact, soc_id, pte_path, inputs, adb)


def generate_htp_profile_result(
    artifact_dir: str,
    soc_id: QcomChipset,
    pte_path: str,
    inputs: Sequence[Tuple[torch.Tensor]],
    adb,
) -> List[QnnHtpProfileArtifacts]:
    """Generate HTP optrace artifacts from a .pte by running on device.

    Arguments:
    - artifact_dir: host directory for dumped `.dlc`/`.bin`, schematics, QNN
      configs, profiling logs, and qnn-profile-viewer outputs.
    - soc_id: target SoC used by the compiled `.pte`; must match the device.
    - pte_path: `.pte` produced by build_executorch_binary().
    - inputs: sample input tensors used by qnn-net-run for optrace collection.
    - adb: SimpleADB helper for pushing files, running qnn-net-run, and
      pulling `qnn-profiling-data_0.log` from the device.

    Supported prepare modes for the input pte file:
    - `QnnConfig.online_prepare=True`: `.pte` contains `.dlc`;
      We will create the context binary and schematic before device execution.
    - `QnnConfig.online_prepare=False`: `.pte` contains finalized `.bin`, user must
      also set `QnnConfig.profile_level=3` so optrace instrumentation is already baked in;
      We will extract schematic and context binary from pte and continue device execution.

    """
    return _generate_htp_analysis_result(
        artifact_dir=artifact_dir,
        soc_id=soc_id,
        pte_path=pte_path,
        inputs=inputs,
        mode="optrace",
        adb=adb,
    )


def estimate_htp_profile_result(
    artifact_dir: str,
    soc_id: QcomChipset,
    pte_path: str,
) -> List[QnnHtpProfileArtifacts]:
    """Estimate HTP performance with host-only hextimate artifacts.

    Arguments:
    - artifact_dir: host directory for dumped `.dlc`, QNN configs, and
      qnn-profile-viewer outputs.
    - soc_id: target SoC used by the compiled `.pte`; currently limited to:
      SA8540, SA8255, QCS9100, and SA8797.
    - pte_path: `.pte` produced by build_executorch_binary(),  requiring
    `QnnConfig.online_prepare=True` for `.pte` generation and QNN SDK >= 2.41.
    """
    _validate_hextimate_soc(soc_id)
    return _generate_htp_analysis_result(
        artifact_dir=artifact_dir,
        soc_id=soc_id,
        pte_path=pte_path,
        mode="hextimate",
    )
