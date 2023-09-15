# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping

from executorch.sdk.edir.et_schema import FXOperatorGraph, OperatorGraphWithStats
from executorch.sdk.etrecord import ETRecord


# TODO: add a unittest for this function
def gen_graphs_from_etrecord(
    etrecord: ETRecord,
) -> Mapping[str, OperatorGraphWithStats]:
    if etrecord.graph_map is None:
        return {}
    return {
        name: FXOperatorGraph.gen_operator_graph(exported_program.graph_module)
        for name, exported_program in etrecord.graph_map.items()
    }
