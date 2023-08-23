# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Mapping, Optional

from executorch.sdk.edir.et_schema import InferenceRun, OperatorGraphWithStats
from pandas import DataFrame


class Inspector:

    """
    APIs for examining model architecture and performance stats
    """

    def __init__(
        self,
        op_graph_dict: Mapping[str, OperatorGraphWithStats],
        show_stack_trace: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Constructor that returns a Debugger instance from a dict of OperatorGraph instances and some optional parameters
        """
        # Save the parameters into class members
        self.op_graph_dict = op_graph_dict
        self.show_stack_trace = show_stack_trace
        self.verbose = verbose

        # Construct the initial tables and save in class members
        self.architecture_table = self._gen_architecture_table()
        self.high_level_profile_info_table = self._gen_high_level_profile_info_table()

    def attach_etdump(self, etdump_path: str) -> None:
        """
        API that attaches ETDump to this inspector instance
        """
        op_graph = self.op_graph_dict["et_dialect_graph_module/forward"]

        if os.path.exists(etdump_path):
            op_graph.attach_metadata(
                inference_run=InferenceRun.extract_runs_from_path(
                    file_path=etdump_path
                )[0]
            )
        else:
            raise Exception("Invalid ET Dump path")

    def get_high_level_profile_info_table(self) -> DataFrame:
        """
        API that returns the high level profile information table from class member
        """
        return self.high_level_profile_info_table

    def get_architecture_table(self) -> DataFrame:
        """
        API that returns the architecture table from class member, filtered by user options (e.g. self.show_stack_trace)
        """
        # TODO: filter based on user options (self.show_stack_trace) before return
        return self.architecture_table

    def cli_flow(self):
        """
        API that enters the CLI debugging flow
        """
        print("Entering the CLI debugging flow...")

        print("High level profile information table:")
        print(self.get_high_level_profile_info_table())
        print("Architecture table:")
        print(self.get_architecture_table())

        # TODO: Take user commands, process by calling other APIs in the Inspector class

    def select_by_instance_id(self, instance_id: str) -> DataFrame:
        """
        API that returns a DataFrame containing information for a specific instance id
        """
        # TODO: filter the architecture table by the instance id before return
        return self.architecture_table

    def select_by_instance_type(self, instance_type: str) -> DataFrame:
        """
        API that returns a DataFrame containing instances with the given instance type
        """
        # TODO: filter the architecture table by the instance type before return
        return self.architecture_table

    def _gen_high_level_profile_info_table(self) -> DataFrame:
        """
        Private helper function that generates the high level profile information table
        """
        # TODO: implement
        pass

    def _gen_architecture_table(self) -> DataFrame:
        """
        Private helper function that generates the architecture table
        """
        # TODO: implement
        pass
