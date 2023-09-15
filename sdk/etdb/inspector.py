# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from executorch.exir import ExportedProgram

from executorch.sdk.edir.et_schema import OperatorGraphWithStats
from executorch.sdk.etdb._inspector_utils import gen_graphs_from_etrecord
from executorch.sdk.etrecord import parse_etrecord
from tabulate import tabulate


@dataclass
class PerfData:
    def __init__(self, raw: List[float]):
        self.raw: List[float] = raw

    @property
    def p50(self) -> float:
        return np.percentile(self.raw, 50)

    @property
    def p90(self) -> float:
        return np.percentile(self.raw, 90)

    @property
    def avg(self) -> float:
        return np.mean(self.raw)

    @property
    def min(self) -> float:
        return min(self.raw)

    @property
    def max(self) -> float:
        return max(self.raw)

    @property
    def median(self) -> float:
        return np.median(self.raw)


# TODO: detailed documentation
@dataclass
class Event:
    """
    Corresponds to an op instance
    """

    name: str
    perf_data: PerfData
    op_type: List[str] = dataclasses.field(default_factory=list)

    # Instruction Id of the original profiling event
    instruction_id: Optional[int] = None

    # Supplemental Identifier used in combination with instruction_identifier
    delegate_debug_identifier: Optional[Union[int, str]] = None

    # Debug Handles in the model graph to which this event is correlated
    debug_handles: Optional[Union[int, List[int]]] = None

    stack_trace: Dict[str, str] = dataclasses.field(default_factory=dict)
    module_hierarchy: Dict[str, Dict] = dataclasses.field(default_factory=dict)
    is_delegated_op: Optional[bool] = None
    delegate_backend_name: Optional[str] = None
    debug_data: List[torch.Tensor] = dataclasses.field(default_factory=list)


@dataclass
class EventBlock:
    """
    EventBlock contains a collection of events associated with a particular profiling/debugging block retrieved from the runtime.
    Attributes:
        name (str): Name of the profiling/debugging block
        events (List[Event]): List of events associated with the profiling/debugging block
    """

    name: str
    events: List[Event] = dataclasses.field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the EventBlock into a DataFrame with each row being an event instance
        """
        # TODO: push row generation down to Event
        data = {
            "event_block_name": [self.name] * len(self.events),
            "event_name": [event.name for event in self.events],
            "raw": [event.perf_data.raw for event in self.events],
            "p50": [event.perf_data.p50 for event in self.events],
            "p90": [event.perf_data.p90 for event in self.events],
            "avg": [event.perf_data.avg for event in self.events],
            "min": [event.perf_data.min for event in self.events],
            "max": [event.perf_data.max for event in self.events],
            "median": [event.perf_data.median for event in self.events],
            "op_type": [event.op_type for event in self.events],
            "delegate_debug_identifier": [
                event.delegate_debug_identifier for event in self.events
            ],
            "stack_traces": [event.stack_trace for event in self.events],
            "module_hierarchy": [event.module_hierarchy for event in self.events],
            "is_delegated_op": [event.is_delegated_op for event in self.events],
            "delegate_backend_name": [
                event.delegate_backend_name for event in self.events
            ],
            "debug_data": [event.debug_data for event in self.events],
        }
        df = pd.DataFrame(data)
        return df


class Inspector:
    """
    APIs for examining model architecture and performance stats
    """

    def __init__(
        self, etdump_path: Optional[str] = None, etrecord_path: Optional[str] = None
    ) -> None:
        """
        Create an inspector instance from the provided ETDump/ETRecord
        """

        # Gen op graphs from etrecord
        if etrecord_path is not None:
            self._etrecord = parse_etrecord(etrecord_path=etrecord_path)
            self._op_graph_dict: Mapping[
                str, OperatorGraphWithStats
            ] = gen_graphs_from_etrecord(etrecord=self._etrecord)

        self.event_blocks: List[EventBlock] = []
        # TODO: create event blocks from etdump, and associate events with op graph nodes

    def print_data_tabular(self) -> None:
        """
        Prints the underlying EventBlocks (essentially all the performance data)
        """

        def style_text_size(val, size=12):
            return f"font-size: {size}px"

        df_list = [event_block.to_dataframe() for event_block in self.event_blocks]
        combined_df = pd.concat(df_list, ignore_index=True)
        # TODO: filter out raw, delegate_debug_identifier, stack_traces and module_hierarchy
        try:
            from IPython.display import display

            styled_df = combined_df.style.applymap(style_text_size)
            display(styled_df)
        except:
            print(tabulate(combined_df, headers="keys", tablefmt="fancy_grid"))

    def get_event_blocks(self) -> List[EventBlock]:
        """
        Returns EventBlocks containing contents from ETDump (correlated to ETRecord if provided)
        """
        return self.event_blocks

    def get_op_list(
        self, event_block: str, show_delegated_ops: Optional[bool] = True
    ) -> Dict[str, List[Event]]:
        """
        Return a map of op_types to Events of that op_type
        """
        # TODO: implement
        return {}

    def write_tensorboard_artifact(self, path: str) -> None:
        """
        Write to the provided path, the artifacts required for visualization in TensorBoard
        """
        # TODO: implement
        pass

    # TODO: add a unittest for this function
    def get_exported_program(self, graph: Optional[str]) -> ExportedProgram:
        """
        Access helper for ETRecord, defaults to returning Edge Dialect Program
        """
        if not graph:
            return self._etrecord["edge_dialect_output/forward"]
        else:
            return self._etrecord.get(graph)
