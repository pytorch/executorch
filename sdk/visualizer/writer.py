from typing import Dict, Tuple

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from torch.utils.tensorboard import SummaryWriter


class ETWriter(SummaryWriter):
    def add_graph_and_tagged_run_metadata(
        self,
        graph_profile: Tuple[GraphDef, Dict[str, RunMetadata]],
        walltime=None,
    ) -> None:
        """Adds a `Graph` and associated `RunMetadata`(s) protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and a dict of `RunMetadata` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
                    walltime (from time.time()) seconds after epoch
        """
        graph = graph_profile[0]
        run_metadata_dict = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        file_writer = self._get_file_writer()
        file_writer.add_event(event, None, walltime)

        for key, value in run_metadata_dict.items():
            trm = event_pb2.TaggedRunMetadata(
                tag=key, run_metadata=value.SerializeToString()
            )
            event = event_pb2.Event(tagged_run_metadata=trm)
            file_writer.add_event(event, None, walltime)
