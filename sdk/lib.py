import argparse
import asyncio
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional

from executorch.sdk.edir.et_schema import FXOperatorGraph, InferenceRun
from executorch.sdk.etdb.etdb import debug_graphs
from executorch.sdk.etrecord import ETRecord, generate_etrecord, parse_etrecord  # noqa

from executorch.sdk.visualizer.generator import Generator
from manifold.clients.python import ManifoldClient

"""
Private Lib Helpers
"""


def _gen_graphs_from_etrecord(etrecord: ETRecord) -> Dict[str, FXOperatorGraph]:
    if etrecord.graph_map is None:
        return {}
    return {
        name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
        for name, exported_program in etrecord.graph_map.items()
    }


async def _gen_and_attach_metadata(
    op_graph_dict: Dict[str, FXOperatorGraph], et_dump_path: str
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

    if os.path.exists(et_dump_path):  # Local path
        op_graph.attach_metadata(
            inference_run=InferenceRun.extract_runs_from_path(file_path=et_dump_path)[0]
        )
    elif et_dump_path.startswith("//manifold/"):  # Manifold path
        et_dump_path = et_dump_path.replace("//manifold/", "", 1)
        et_dump_bucket, et_dump_blob = et_dump_path.split("/", 1)
        # Download to a temp local path
        with tempfile.TemporaryDirectory() as tmpdir, ManifoldClient.get_client(
            et_dump_bucket
        ) as client:
            et_dump_path_local = os.path.join(tmpdir, et_dump_bucket, et_dump_blob)
            os.makedirs(os.path.dirname(et_dump_path_local), exist_ok=True)
            client.sync_get(path=et_dump_blob, output=et_dump_path_local)

            op_graph.attach_metadata(
                inference_run=InferenceRun.extract_runs_from_path(
                    file_path=et_dump_path_local
                )[0]
            )
    else:  # Invalid path
        raise Exception("Invalid ET Dump path")


"""
SDK Entry Points
"""


async def debug_etrecord(
    etrecord: ETRecord, et_dump_path: Optional[str] = None, verbose: bool = False
):
    """
    Given an ETRecord, kick off ETDB
    """
    op_graph_dict: Dict[str, FXOperatorGraph] = _gen_graphs_from_etrecord(etrecord)
    if et_dump_path is not None:
        await _gen_and_attach_metadata(op_graph_dict, et_dump_path)
    # pyre-ignore Expecting value to be a Union, but provided arg with one type
    debug_graphs(op_graph_dict, verbose)


async def debug_etrecord_path(
    etrecord_path: str, et_dump_path: Optional[str] = None, verbose: bool = False
):
    """
    Given a path to an ETRecord, kick off ETDB
    """
    await debug_etrecord(parse_etrecord(etrecord_path), et_dump_path, verbose)


async def visualize_etrecord(
    etrecord: ETRecord,
    et_dump_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """
    Given an ETRecord, visualize it via TensorBoard
    If run_name is None, a unique run name will be generated

    Returns: a TensorBoard On Demand URL
    """
    op_graph_dict: Dict[str, FXOperatorGraph] = _gen_graphs_from_etrecord(etrecord)
    if et_dump_path is not None:
        await _gen_and_attach_metadata(op_graph_dict, et_dump_path)

    generator = Generator()
    return await generator.gen_multiple(
        # pyre-fixme[[6]]: Incompatible parameter type [6]: In call `Generator.gen_multiple`, for argument `op_graphs_dict`,
        # expected `Dict[str, OperatorGraph]` but got `Dict[str, Union[ExportedETOperatorGraph, FXOperatorGraph]]`.
        op_graphs_dict=op_graph_dict,
        run_name=run_name
        if run_name
        else "sdk_e2e_" + datetime.now().strftime("%b%d_%H-%M-%S"),
    )


async def visualize_etrecord_path(
    etrecord_path: str,
    et_dump_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """
    See visualize_etrecord for more details

    Given an path to an ETRecord, visualize it via TensorBoard
    If run_name is None, a unique run name will be generated

    Returns: a TensorBoard On Demand URL
    """
    return await visualize_etrecord(
        parse_etrecord(etrecord_path), et_dump_path, run_name
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("et_record", help="Path to ETRecord")
    parser.add_argument("--et_dump", help="Path to ET Dump")
    parser.add_argument("--run_name", help="Unique name of the TB run")
    parser.add_argument(
        "--terminal_mode", action="store_true", help="Use a terminal to debug"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether the terminal should display in verbose mode",
    )
    return parser.parse_args()


async def main() -> int:
    """
    Simple CLI wrapper for calling either
      - debug_etrecord_path (terminal_mode)
      - visualize_etrecord_path

    Only required argument is an et_record path
    """
    args = parse_args()

    if args.terminal_mode:
        await debug_etrecord_path(args.et_record, args.et_dump, args.verbose)
    else:
        print(
            await visualize_etrecord_path(args.et_record, args.et_dump, args.run_name)
        )

    return 0


if __name__ == "__main__":
    asyncio.run(main())
