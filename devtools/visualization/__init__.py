# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# model_explorer-based visualization (requires model_explorer pip package)
try:
    from executorch.devtools.visualization.visualization_utils import (  # noqa: F401
        ModelExplorerServer,
        SingletonModelExplorerServer,
        visualize,
        visualize_graph,
        visualize_model_explorer,
    )
except ImportError:
    pass

# Self-contained HTML visualization (no external dependencies)
from executorch.devtools.visualization.html_visualization import (  # noqa: F401
    extract_from_exported_program,
    extract_from_etrecord,
    extract_from_pt2,
    extract_from_pte,
    generate_html,
    generate_multi_pass_html,
    visualize_edge_manager,
)
