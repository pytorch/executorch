# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import cast, Optional, Tuple

import torch
from executorch.devtools import Inspector
from executorch.devtools.inspector import Event, EventBlock, PerfData
from executorch.devtools.inspector._inspector_utils import TimeScale
from tabulate import tabulate


class CadenceETDump:
    def __init__(self, output_dir: str) -> None:
        self.tensor_dump_dir: str = os.path.join(output_dir, "tensors")
        self.etdump_path: str = os.path.join(output_dir, "etdump.etdp")
        self.etrecord_path: Optional[str] = os.path.join(output_dir, "etrecord.bin")
        self.debug_buffer_path: Optional[str] = os.path.join(
            output_dir, "debug_output.bin"
        )

        if not os.path.exists(self.etdump_path):
            raise RuntimeError(f"{self.etdump_path} does not exist")
        # pyre-ignore[6]: os.path.exists expects str, but got Optional[str]
        if not os.path.exists(self.etrecord_path):
            logging.warning(
                "ETRecord not found, intermediate tensors will not be dumped"
            )
            self.etrecord_path = None
        # pyre-ignore[6]: os.path.exists expects str, but got Optional[str]
        if not os.path.exists(self.debug_buffer_path):
            logging.warning(
                "Debug buffer not found, intermediate tensors will not be dumped"
            )
            self.debug_buffer_path = None

        self.et_inspector: Inspector = Inspector(
            etdump_path=self.etdump_path,
            debug_buffer_path=self.debug_buffer_path,
            etrecord=self.etrecord_path,
            source_time_scale=TimeScale.CYCLES,
            target_time_scale=TimeScale.CYCLES,
        )

    def get_outputs(self, log_to_stdout: bool = False) -> Tuple[torch.Tensor]:
        output = [
            event_block.run_output
            for event_block in self.et_inspector.event_blocks
            if event_block.name == "Execute"
        ]
        logging.debug(f"[CadenceETDump] output: {output}")
        return output[0]

    def get_execute_event_block(self) -> EventBlock:
        exec_blocks = [
            eb for eb in self.et_inspector.event_blocks if eb.name == "Execute"
        ]
        return exec_blocks[0]

    def should_include_event(self, event: Event) -> bool:
        # exclude duplicate events
        if event.name in ("OPERATOR_CALL", "Method::execute"):
            return False

        # exclude custom multi-zion events
        if event.name.startswith("DELEGATE_ZION"):
            return False

        return True

    def print_summary(
        self,
        bundled_prog_size: Optional[int] = None,
        external_link: Optional[str] = None,
    ) -> None:
        """
        Print performance summary with optional program size and external link.

        Args:
            bundled_prog_size: Size of the bundled program in bytes (optional)
            external_link: External analytics/monitoring link (optional, e.g., Scuba link for Meta internal use)
        """
        block = self.get_execute_event_block()
        op_events = [e for e in block.events if self.should_include_event(e)]
        op_time_sum = sum([cast(PerfData, e.perf_data).avg for e in op_events])

        overall_event = [ev for ev in block.events if ev.name == "Method::execute"]
        if not len(overall_event) == 1:
            logging.warning(
                f"Expected one 'Method::execute' event, found {len(overall_event)}"
            )

        total_cycles = cast(PerfData, overall_event[0].perf_data).avg
        op_cycles = op_time_sum

        # Build table data and headers dynamically based on what's provided
        table_data = [
            "{:,.0f}".format(total_cycles),
            "{:,.0f}".format(op_cycles),
            "{:,.0f}".format(total_cycles - op_cycles),
            "{:.2%}".format((total_cycles - op_cycles) / total_cycles),
        ]
        headers = [
            "Total Cycles",
            "Cycles in Ops",
            "Other Cycles",
            "Framework Tax (%)",
        ]

        # Add optional fields if provided
        if bundled_prog_size is not None:
            table_data.append("{:,.0f}".format(bundled_prog_size))
            headers.append("Bundled Program Size (bytes)")

        if external_link is not None:
            table_data.append(external_link)
            headers.append("External Link")

        logging.info(
            "Performance Summary:\n%s",
            tabulate(
                [table_data],
                headers=headers,
                tablefmt="outline",
            ),
        )

    def print_event_block(self) -> None:
        logging.info("Profiled events:")
        if logging.getLogger().level <= logging.INFO:
            self.et_inspector.print_data_tabular()

    def dump_intermediate_tensors(self) -> None:
        if self.etrecord_path is None:
            logging.info("[CadenceETDump] Intermediate tensors not available")
            return

        logging.info(
            f"[CadenceETDump] Dumping intermediate tensors to {self.tensor_dump_dir}"
        )
        os.makedirs(self.tensor_dump_dir, exist_ok=True)
        exec_blocks = [
            eb for eb in self.et_inspector.event_blocks if eb.name == "Execute"
        ]
        if len(exec_blocks) > 1:
            logging.warning(
                f'Found {len(exec_blocks)} "Execute" blocks, using the first one and ignoring the rest.'
            )
        block = exec_blocks[0]

        # OPERATOR_CALL events are duplicates that contain framework tax data. We don't need them
        op_events = [e for e in block.events if e.name != "OPERATOR_CALL"]
        torch.set_printoptions(profile="full")

        for event in op_events:
            instr_id = event._instruction_id
            if not event.debug_data:
                logging.debug(
                    f"Missing intermediate tensor data for {event.name} ({instr_id=})"
                )
                continue

            with open(f"{self.tensor_dump_dir}/{instr_id}.txt", "w") as f:
                for dd in event.debug_data:
                    f.write(f"{str(dd)}\n\n")
        torch.set_printoptions(profile="default")
