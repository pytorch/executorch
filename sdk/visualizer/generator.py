import asyncio
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

from executorch.sdk.edir.et_schema import OperatorGraph, RESERVED_METADATA_ARG

from executorch.sdk.visualizer.converter import Converter
from executorch.sdk.visualizer.utils import make_markdown_table
from executorch.sdk.visualizer.writer import ETWriter
from manifold.clients.python import ManifoldClient
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef


MANIFOLD_BUCKET = "tensorboard_logs_etdb"
MANIFOLD_NAMESPACE = "tree"
LOG_DIR_BASE = "runs"
TB_OD_URL_BASE = (
    "https://www.internalfb.com/intern/tensorboard/?tier={}&dir=manifold://{}"
)
ET_TB_SMC_TIER = "tensorboard.et.prod_https"


class Generator:
    """
    Provides the gen() API to generate TensorBoard URLs
    """

    async def gen_multiple(
        self,
        op_graphs_dict: Dict[str, OperatorGraph],
        run_name: str,
    ) -> str:
        """
        Generates a TensorBoard on demand URL for a hosted TB page that visualizes the input op_graphs.

        args:
            op_graphs_dict (Dict[str, OperatorGraph]): OperatorGraphs to visualize, keyed by graph name
            run_name (str): A unique identifier for this run to be used by TensorBoard

        returns: a TensorBoard On Demand URL of visualizations for all graphs under the op_graphs_dict
        """
        coroutines = [
            self.gen(op_graph, run_name, graph_name)
            for graph_name, op_graph in op_graphs_dict.items()
        ]
        urls = await asyncio.gather(*coroutines)

        # All the urls in the list are the same, so just return the first one
        return urls[0]

    async def gen(
        self,
        op_graph: OperatorGraph,
        run_name: str,
        graph_name: Optional[str] = None,
    ) -> str:
        """
        Generates a TensorBoard on demand URL for a hosted TB page that visualizes the input op_graph.

        args:
            op_graph (OperatorGraph): OperatorGraph to visualize
            run_name (str): A unique identifier for this run to be used by TensorBoard
                            If not provided, will use current time as the identifier
            graph_name (str): Optional identifier for this graph under run_name

        returns: a TensorBoard On Demand URL
        """

        # Call the edir->tb format converter
        converter = Converter(op_graph=op_graph)
        graph_profile = converter.convert()

        # Call TB writer and upload log file
        log_run_dir = f"{LOG_DIR_BASE}/{run_name}"

        # Use run_name as graph_name when graph_name is not provided
        graph_name = graph_name or run_name
        await self._write_and_upload(
            graph_profile=graph_profile,
            run_level_metadata=op_graph.metadata or {},
            log_dir=f"{log_run_dir}/{graph_name}",
        )

        # Construct and return the TB On Demand URL
        manifold_dir = f"{MANIFOLD_BUCKET}/{MANIFOLD_NAMESPACE}/{log_run_dir}"
        tb_od_url = TB_OD_URL_BASE.format(ET_TB_SMC_TIER, manifold_dir)
        return tb_od_url

    async def _write_and_upload(
        self,
        graph_profile: Tuple[GraphDef, RunMetadata],
        run_level_metadata: Dict[str, Any],
        log_dir: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tmp_log_dir = os.path.join(tmpdir, log_dir)
            summary_writer = ETWriter(log_dir=local_tmp_log_dir)

            # Call the add_graph function that takes a TB format object describing a graph and its run stats
            summary_writer.add_graph_and_tagged_run_metadata(
                # pyre-fixme[6]: For 1st argument expected `Tuple[GraphDef,
                #  Dict[str, RunMetadata]]` but got `Tuple[GraphDef, RunMetadata]`.
                graph_profile=graph_profile
            )

            # Call the add_text function to add run level stats to the TB log which will
            # get rendered under the "Text" tab
            for key, value in run_level_metadata.items():
                if key == RESERVED_METADATA_ARG.TABLES_KEYWORD.value:
                    for table_name, table_content in value.items():
                        summary_writer.add_text(
                            table_name, make_markdown_table(table=table_content)
                        )
                elif key == RESERVED_METADATA_ARG.KV_KEYWORD.value:
                    for k, v in value.items():
                        summary_writer.add_text(k, str(v))

            # This closes the file_writer as well
            summary_writer.close()

            # Upload the TB log file from local to Manifold
            file = os.listdir(local_tmp_log_dir)[0]
            with ManifoldClient.get_client(MANIFOLD_BUCKET) as client:
                await client.mkdir(
                    path=f"{MANIFOLD_NAMESPACE}/{log_dir}", recursive=True
                )
                await client.put(
                    f"{MANIFOLD_NAMESPACE}/{log_dir}/{file}",
                    f"{local_tmp_log_dir}/{file}",
                )
