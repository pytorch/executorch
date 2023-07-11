import argparse
import asyncio
import os
import tempfile
from datetime import datetime
from enum import Enum
from typing import Dict, Union

from executorch.sdk.edir.et_schema import (
    ExportedETOperatorGraph,
    FXOperatorGraph,
    InferenceRun,
)
from executorch.sdk.etdb.etdb import debug_graph
from executorch.sdk.etrecord import ETRecord, parse_etrecord
from executorch.sdk.visualizer.generator import Generator
from manifold.clients.python import ManifoldClient

# Keywords used to identify graphs
class GRAPH_NAME(Enum):
    FORWARD = "forward"
    ET_DIALECT_FORWARD = "et_dialect_graph_module/forward"


async def gen_op_graph_from_program(program_path: str) -> ExportedETOperatorGraph:
    """
    Deserialize the program under program_path and construct an ETDF operator graph from it

    Args:
        program_path (str): local or Manifold path to the Program to be visualized

    Returns: an ETDF operator graph object
    """
    if os.path.exists(program_path):  # Local path
        return ExportedETOperatorGraph.gen_operator_graph_from_path(
            file_path=program_path
        )
    elif program_path.startswith("//manifold/"):  # Manifold path
        program_path = program_path.replace("//manifold/", "", 1)
        program_bucket, program_blob = program_path.split("/", 1)
        # Download to a temp local path
        with tempfile.TemporaryDirectory() as tmpdir, ManifoldClient.get_client(
            program_bucket
        ) as client:
            program_path_local = os.path.join(tmpdir, program_bucket, program_blob)
            os.makedirs(os.path.dirname(program_path_local), exist_ok=True)
            client.sync_get(path=program_blob, output=program_path_local)

            return ExportedETOperatorGraph.gen_operator_graph_from_path(
                file_path=program_path_local
            )
    else:  # Invalid path
        raise Exception("Invalid program path")


async def gen_op_graphs_from_etrecord(etrecord: ETRecord) -> Dict[str, FXOperatorGraph]:
    # TODO : We don't support multiple entry points yet, assert until we do.
    graph_map = etrecord.graph_map
    assert graph_map is not None, "ETRecord missing graph modules to be visualized."

    op_graph_dict = {
        name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
        for name, exported_program in graph_map.items()
    }
    return op_graph_dict


async def gen_and_attach_metadata(
    op_graph: Union[FXOperatorGraph, ExportedETOperatorGraph], et_dump_path: str
) -> None:
    """
    Attach metadata in ETDump under path et_dump_path to the given op_graph.
    To visualize op_graph without ETDump metadata, this function can be skipped.

    Args:
        op_graph (ExportedETOperatorGraph): operator graph to visualize
        et_dump_path (str): local or Manifold path to the ETDump
    """

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


async def gen_tb_url(
    op_graphs_dict: Dict[str, Union[FXOperatorGraph, ExportedETOperatorGraph]],
    run_name: str,
) -> str:
    """
    Generate a link to TensorBoard visualizing the given operator graph

    Args:
        op_graph (ExportedETOperatorGraph): operator graph to visualize
        run_name (str): Unique name of the run to be displayed on TensorBoard. This has to be unique because a log file
                        will be created under this name in a manifold bucket that has been onboarded to Tensorboard On Demand

    Returns: a TensorBoard On Demand URL
    """
    # Initialize the TB URL generator and call gen()
    generator = Generator()
    return await generator.gen_multiple(
        # pyre-fixme[[6]]: Incompatible parameter type [6]: In call `Generator.gen_multiple`, for argument `op_graphs_dict`,
        # expected `Dict[str, OperatorGraph]` but got `Dict[str, Union[ExportedETOperatorGraph, FXOperatorGraph]]`.
        op_graphs_dict=op_graphs_dict,
        run_name=run_name,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", help="Path to Model Flatbuffer")
    parser.add_argument("--etrecord", help="Path to ETRecord")
    parser.add_argument("--et_dump", help="ET Dump")
    parser.add_argument("--run_name", help="Unique name of this run")
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
    args = parse_args()

    if args.program is None and args.etrecord is None:
        raise Exception("Either --program or --etrecord must be specified")

    if args.program is not None and args.etrecord is not None:
        raise Exception("Only one of --program or --etrecord can be specified")

    op_graph_dict = {}
    if args.program is not None:
        op_graph = await gen_op_graph_from_program(program_path=args.program)
        if args.et_dump is not None:
            await gen_and_attach_metadata(op_graph=op_graph, et_dump_path=args.et_dump)
        op_graph_dict[GRAPH_NAME.FORWARD.value] = op_graph
    elif args.etrecord is not None:
        etrecord = parse_etrecord(args.etrecord)
        op_graph_dict = await gen_op_graphs_from_etrecord(etrecord)
        # Currently we only support attaching etdump data to the et_dialect_graph_module.
        if args.et_dump is not None:
            await gen_and_attach_metadata(
                op_graph=op_graph_dict[GRAPH_NAME.ET_DIALECT_FORWARD.value],
                et_dump_path=args.et_dump,
            )

    assert op_graph_dict, "Failed to generate graph for visualization."

    if args.terminal_mode:
        if args.program is not None:
            debug_graph(op_graph_dict[GRAPH_NAME.FORWARD.value], args.verbose)
        elif args.etrecord is not None:
            debug_graph(
                op_graph_dict[GRAPH_NAME.ET_DIALECT_FORWARD.value], args.verbose
            )
        exit()

    default_run_name = "sdk_e2e_" + datetime.now().strftime("%b%d_%H-%M-%S")
    tb_url = await gen_tb_url(
        op_graphs_dict=op_graph_dict,
        run_name=args.run_name or default_run_name,
    )

    # Print the returned URL
    print(f"\nTo view the graph of this run, go to: {tb_url}\n")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
