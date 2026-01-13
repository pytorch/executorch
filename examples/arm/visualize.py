# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import io
import json
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path

from typing import Any, Callable, Dict, Iterable, NamedTuple, Union

import pandas as pd

from executorch.devtools.visualization.visualization_utils import (
    visualize_model_explorer,
)
from model_explorer import config as model_explorer_config, node_data_builder as ndb
from model_explorer.config import ModelSource

COMPILER_OP_ID = "scheduled_id"


class Tables(NamedTuple):
    queue: pd.DataFrame
    group: pd.DataFrame
    perf: pd.DataFrame
    source: pd.DataFrame


def parse_tables(tables_path: Path) -> Tables:
    """
    Parse the XML debug tables file and extract required tables as pandas DataFrames.
    """
    required_tables = {"queue", "group", "perf", "source"}
    try:
        tree = ET.parse(tables_path)  # nosec B314
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML tables file {tables_path}: {e}")

    tables: Dict[str, pd.DataFrame] = {}
    for table in tree.getroot().findall("table"):
        name = table.attrib.get("name")
        if name in required_tables:
            text = table.text or ""
            tables[name] = pd.read_csv(io.StringIO(text))

    missing = required_tables - tables.keys()
    if missing:
        raise ValueError(f"Missing required tables in XML: {missing}")

    return Tables(**tables)


def get_trace_file_objects(trace_file_path: Path) -> list[Dict[str, Any]]:
    """
    Load and return the 'traceEvents' list from a gzip-compressed JSON trace file.
    """
    try:
        with gzip.open(trace_file_path, "rt", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read or parse trace file {trace_file_path}: {e}")

    if "traceEvents" not in data:
        raise KeyError(f"'traceEvents' key not found in {trace_file_path}")

    return data["traceEvents"]


def get_subops(df_group: pd.DataFrame) -> set:
    return set(df_group[df_group["id"] != df_group["group_id"]]["id"])


def transform_events(
    objects: Iterable[Dict[str, Any]], queue_df: pd.DataFrame, sub_ops: set
) -> None:
    """
    Annotate the 'queue' table in-place with duration based on trace events.
    """
    queue_df_len = len(queue_df)
    offsets = queue_df["offset"].astype(int)

    start_ts, cmd_index, chain_len = 0, 0, 1

    def is_end_of_command(qread_offset: int, end_idx: int) -> bool:
        if end_idx >= queue_df_len:
            return qread_offset > offsets[cmd_index]
        return qread_offset == offsets[end_idx]

    for event in (e for e in objects if e.get("tid") == "qread"):
        if cmd_index >= queue_df_len:
            break

        qread_offset = 4 * int(event["args"]["qread"])

        while (cmd_index + chain_len <= queue_df_len - 1) and queue_df.iloc[
            cmd_index + chain_len
        ]["scheduled_id"] in sub_ops:
            chain_len += 1

        end_idx = cmd_index + chain_len
        if is_end_of_command(qread_offset, end_idx):
            end_ts = int(event["ts"]) - 1
            queue_df.loc[cmd_index, ["duration"]] = [
                end_ts - start_ts,
            ]
            start_ts = end_ts
            cmd_index = end_idx
            chain_len = 1


Agg = Union[str, Callable[[pd.Series], Any]]


def list_unique(s: pd.Series) -> list[Any]:
    return sorted(set(s.dropna()))


def build_perf_df(tables: Tables) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a performance DataFrame summarizing queue metrics grouped by source_id.
    Returns a tuple of (perf_df, cmd_to_op_df) where cmd_to_op_df is needed for unmapped op tracking.
    """
    tables.queue["cmd_id"] = tables.queue.index

    excluded = {"optimised_id", "scheduled_id", "offset"}
    col_funcs: Dict[str, Agg] = {
        c: "sum" for c in tables.queue.columns if c not in excluded
    }

    col_funcs.update({"cmdstream_id": list_unique, "cmd_id": list_unique})

    cmd_to_op_df = tables.queue.groupby(COMPILER_OP_ID).agg(col_funcs).reset_index()

    opt_df = (
        pd.merge(tables.perf[["id", "source_id"]], tables.group, on="id", how="left")
        .rename(columns={"id": COMPILER_OP_ID})
        .merge(cmd_to_op_df, on=COMPILER_OP_ID, how="inner")
    )

    exclude_columns = ["source_id"]
    src_col_funcs: Dict[str, Agg] = {
        col: "sum" for col in opt_df.columns if col not in exclude_columns
    }
    src_col_funcs[COMPILER_OP_ID] = list_unique

    perf_df = opt_df.groupby("source_id").agg(src_col_funcs).reset_index()

    return perf_df, cmd_to_op_df


def check_unmapped_ops(
    tables: Tables, src_df: pd.DataFrame, cmd_to_op_df: pd.DataFrame
) -> None:
    """
    Identify operators in the performance data that are not mapped to any source operation.
    """
    opt_ids_in_src_table = set()
    for opt_ids in src_df[COMPILER_OP_ID].dropna():
        if type(opt_ids) is list:
            opt_ids_in_src_table.update(opt_ids)

    opt_df = pd.merge(
        tables.perf[["id", "source_id"]], tables.group, on="id", how="left"
    )
    opt_df = opt_df.rename(columns={"id": COMPILER_OP_ID})
    opt_df = pd.merge(opt_df, cmd_to_op_df, on=COMPILER_OP_ID, how="inner")

    unmapped_operators = opt_df[
        ~opt_df[COMPILER_OP_ID].isin(list(opt_ids_in_src_table))
    ]

    if not unmapped_operators.empty:
        print("Warning: There are unmapped operators in the performance data.")
        print(unmapped_operators)
    return None


def build_src_df(tables: Tables, perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge source table with performance metrics and total NPU cycles.
    Returns a tuple of (src_df, cmd_to_op_df) where df_cmd_to_op is needed for unmapped op tracking.
    """
    return pd.merge(
        tables.source.rename(columns={"id": "source_id"})[["ext_key", "source_id"]],
        perf_df,
        on="source_id",
        how="left",
    ).merge(
        tables.perf[["source_id", "npu_cycles"]]
        .groupby("source_id")
        .sum(numeric_only=True)
        .reset_index(),
        on="source_id",
        how="left",
    )


def get_model_node_data(df: pd.DataFrame) -> ndb.ModelNodeData:
    """
    Convert source-level metrics into ModelExplorer node data for duration.
    """
    durations = df["duration"].fillna(0).astype(int)

    duration_results: Dict[str, ndb.NodeDataResult] = {}

    for src, dur in zip(df["ext_key"], durations):
        node_id = f"main/op{int(src)}"
        duration_results[node_id] = ndb.NodeDataResult(value=int(dur))

    gradient = [
        ndb.GradientItem(stop=0.0, bgColor="#ffffff"),
        ndb.GradientItem(stop=0.1, bgColor="#33FF00"),
        ndb.GradientItem(stop=0.2, bgColor="#66FF00"),
        ndb.GradientItem(stop=0.5, bgColor="#FFFF00"),
        ndb.GradientItem(stop=0.7, bgColor="#FF6600"),
        ndb.GradientItem(stop=1.0, bgColor="#FF0000"),
    ]

    return ndb.ModelNodeData(
        graphsData={
            "main": ndb.GraphNodeData(results=duration_results, gradient=gradient)
        }
    )


def build_overlay_data(trace_path: Path, tables_path: Path) -> ndb.ModelNodeData:
    """
    Build ModelExplorer node data from trace and tables files.
    """
    tables = parse_tables(tables_path)
    events = get_trace_file_objects(trace_path)
    transform_events(events, tables.queue, get_subops(tables.group))
    perf_df, cmd_to_op_df = build_perf_df(tables)
    src_df = build_src_df(tables, perf_df)
    check_unmapped_ops(tables, src_df, cmd_to_op_df)

    return get_model_node_data(src_df)


def validate_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")


def validate_perf_mode_args(trace: str, tables: str) -> None:
    if not (trace and tables):
        raise ValueError(
            "Both --trace and --tables must be provided for perf mode, or neither for default mode"
        )


def set_pte_model_explorer_config(model_file, tosa_files, config):
    from pte_adapter_model_explorer.main import PTEAdapter

    pte_adapter = PTEAdapter()

    settings = {"delegate_file_paths": [str(path) for path in tosa_files]}
    me_graphs = pte_adapter.convert(model_path=str(model_file), settings=settings)

    # Convert the given model to model explorer graphs.

    graphs_index = len(config.graphs_list)
    config.graphs_list.append(me_graphs)

    # Construct model source.
    #
    # The model source has a special format, in the form of:
    # graphs://{name}/{graphs_index}
    model_name = model_file.stem
    model_source: ModelSource = {"url": f"graphs://{model_name}/{graphs_index}"}
    config.model_sources.append(model_source)


def set_tosa_model_explorer_config(model_file, config):
    config.add_model_from_path(str(model_file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a model using model explorer."
    )
    parser.add_argument(
        "--model_dir", required=True, type=str, help="Path to the model directory"
    )
    parser.add_argument(
        "--pte",
        action="store_true",
        help="Visualize PTE flatbuffer model and delegates. Cannot be used with --tosa",
    )
    parser.add_argument(
        "--tosa",
        action="store_true",
        help="Visualize TOSA flatbuffer model. Cannot be used with --pte",
    )
    parser.add_argument(
        "--trace",
        required=False,
        help="(perf mode) PMU trace JSON.gz file with performance data. Can only be used together with --tosa",
    )
    parser.add_argument(
        "--tables",
        required=False,
        help="(perf mode) Vela debug database tables XML file",
    )

    args = parser.parse_args()
    if args.pte and args.tosa:
        raise ValueError("Cannot use both --pte and --tosa options together")

    model_dir = Path(args.model_dir).resolve()

    tosa_files = list(model_dir.glob("*TOSA*.tosa"))

    model_files = None
    if args.pte:
        model_files = list(model_dir.glob("*.pte"))
    elif args.tosa:
        model_files = tosa_files

    if not model_files:
        raise FileNotFoundError(
            f"No model files found in {model_dir} for the specified format."
        )

    model_file = model_files[0]

    validate_file_exists(model_file)

    config = model_explorer_config()

    extensions = []

    if args.pte:
        set_pte_model_explorer_config(model_file, tosa_files, config)
    elif args.tosa:
        set_tosa_model_explorer_config(model_file, config)
        extensions.append("tosa_adapter_model_explorer")

    if args.trace or args.tables:
        validate_perf_mode_args(args.trace, args.tables)
        trace_file = Path(args.trace).resolve()
        tables_file = Path(args.tables).resolve()
        validate_file_exists(trace_file)
        validate_file_exists(tables_file)

        config.add_node_data(
            "Duration (Cycles)", build_overlay_data(trace_file, tables_file)
        )

    visualize_model_explorer(config=config, extensions=extensions)


if __name__ == "__main__":
    main()
