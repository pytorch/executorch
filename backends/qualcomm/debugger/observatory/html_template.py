# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import html

from .template_loader import load_css, load_js_chunks


def get_html_template(title: str, payload_json: str) -> str:
    """Generate observatory HTML shell."""

    css = load_css()
    js_bundle = "\n".join(load_js_chunks())

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{html.escape(title)}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div id=\"app\"></div>

    <script>
        window.OBSERVATORY_DATA = {payload_json};
    </script>

    <script>
        if (window.OBSERVATORY_DATA && window.OBSERVATORY_DATA.resources) {{
            const res = window.OBSERVATORY_DATA.resources;
            if (res.css && res.css.length > 0) {{
                const style = document.createElement('style');
                style.textContent = res.css.join('\n');
                document.head.appendChild(style);
            }}
            if (res.js && res.js.length > 0) {{
                const script = document.createElement('script');
                script.textContent = res.js.join(';\n');
                document.body.appendChild(script);
            }}
        }}
    </script>

    <script>
{js_bundle}
    </script>
</body>
</html>"""
