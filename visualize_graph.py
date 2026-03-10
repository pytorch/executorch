#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
"""DEPRECATED: Moved to executorch.devtools.visualization.html_visualization.

Update your imports:
    from executorch.devtools.visualization.html_visualization import (
        visualize_edge_manager,
        generate_html,
    )
"""

import warnings

warnings.warn(
    "visualize_graph has moved to "
    "executorch.devtools.visualization.html_visualization. "
    "Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from executorch.devtools.visualization.html_visualization import (  # noqa: F401
    CATEGORY_COLORS,
    categorize_node,
    extract_from_exported_program,
    extract_from_etrecord,
    extract_from_pt2,
    extract_from_pte,
    extract_from_trace_json,
    generate_html,
    generate_multi_pass_html,
    main,
    visualize_edge_manager,
)

if __name__ == "__main__":
    main()
