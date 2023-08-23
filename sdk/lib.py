# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import os
from typing import Mapping, Optional, Union

from executorch.sdk.edir.et_schema import (
    FXOperatorGraph,
    InferenceRun,
    OperatorGraphWithStats,
)
from executorch.sdk.etdb.etdb import debug_graphs
from executorch.sdk.etdb.inspector import Inspector
from executorch.sdk.etrecord import ETRecord, parse_etrecord

"""
Private Lib Helpers
"""


def _gen_graphs_from_etrecord(
    etrecord: ETRecord,
) -> Mapping[str, OperatorGraphWithStats]:
    if etrecord.graph_map is None:
        return {}
    return {
        name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
        for name, exported_program in etrecord.graph_map.items()
    }


def _gen_and_attach_metadata(
    op_graph_dict: Mapping[str, OperatorGraphWithStats], et_dump_path: str
) -> None:
    """
    (!!) Note: Currently we only support attaching etdump data to the
                et_dialect_graph_module.

    Attach metadata in ETDump under path et_dump_path to the given op_graph.
    To visualize op_graph without ETDump metadata, this function can be skipped.

    Args:
        op_graph (ExportedETOperatorGraph): operator graph to visualize
        et_dump_path (str): local or Manifold path to the ETDump
    """

    op_graph = op_graph_dict["et_dialect_graph_module/forward"]

    if os.path.exists(et_dump_path):
        op_graph.attach_metadata(
            inference_run=InferenceRun.extract_runs_from_path(file_path=et_dump_path)[0]
        )
    else:
        raise Exception("Invalid ET Dump path")


"""
SDK Entry Points
"""


def debug_etrecord(
    etrecord: ETRecord, et_dump_path: Optional[str] = None, verbose: bool = False
):
    """
    Given an ETRecord, kick off ETDB
    """
    op_graph_dict: Mapping[str, OperatorGraphWithStats] = _gen_graphs_from_etrecord(
        etrecord
    )
    if et_dump_path is not None:
        _gen_and_attach_metadata(op_graph_dict, et_dump_path)
    debug_graphs(op_graph_dict, verbose)


def debug_etrecord_path(
    etrecord_path: str, et_dump_path: Optional[str] = None, verbose: bool = False
):
    """
    Given a path to an ETRecord, kick off ETDB
    """
    debug_etrecord(parse_etrecord(etrecord_path), et_dump_path, verbose)


def gen_inspector_from_etrecord(
    etrecord: Union[str, ETRecord],
    etdump_path: Optional[str] = None,
    show_stack_trace: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> Inspector:
    """
    API that creates an Inspector instance based on a file path to an ETRecord instance
    or an ETRecord instance and optional parameters including a file path to an ETDump
    """
    if isinstance(etrecord, str):
        etrecord = parse_etrecord(etrecord_path=str(etrecord))

    op_graph_dict: Mapping[str, OperatorGraphWithStats] = _gen_graphs_from_etrecord(
        etrecord=etrecord
    )
    if etdump_path is not None:
        _gen_and_attach_metadata(op_graph_dict=op_graph_dict, et_dump_path=etdump_path)

    return Inspector(
        op_graph_dict=op_graph_dict, show_stack_trace=show_stack_trace, verbose=verbose
    )


"""
SDK Binary
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("et_record", help="Path to ETRecord")
    parser.add_argument("--et_dump", help="Path to ET Dump")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether the terminal should display in verbose mode",
    )
    parser.add_argument(
        "--show_stack_trace",
        help="Whether to show stack trace in the output tables",
    )
    return parser.parse_args()


async def main() -> int:
    """
    Simple CLI wrapper for triggering ETDB

    Only required argument is an et_record path
    """
    args = parse_args()
    et_inspector = gen_inspector_from_etrecord(
        etrecord=args.et_record,
        etdump_path=args.et_dump,
        show_stack_trace=args.show_stack_trace,
        verbose=args.verbose,
    )
    et_inspector.cli_flow()
    return 0


if __name__ == "__main__":
    asyncio.run(main())
