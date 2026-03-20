# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import html

from .template_loader import load_css, load_js_chunks


def get_html_template(title: str, payload_json: str, is_compressed: bool = False) -> str:
    """Generate observatory HTML shell.

    Args:
        title: Report title shown in <title> and page heading.
        payload_json: Either the raw JSON string (is_compressed=False) or a
            gzip+base64 encoded string (is_compressed=True).
        is_compressed: When True, payload_json is a gzip+base64 blob that the
            browser decompresses via DecompressionStream before parsing.
    """

    css = load_css()
    js_bundle = "\n".join(load_js_chunks())

    if is_compressed:
        data_script = f'window.__OBS_RAW__ = "{payload_json}";'
        decompress_block = """
    async function _obsDecompress(b64gz) {
        const compressed = Uint8Array.from(atob(b64gz), c => c.charCodeAt(0));
        const ds = new DecompressionStream('gzip');
        const writer = ds.writable.getWriter();
        writer.write(compressed);
        writer.close();
        const chunks = [];
        const reader = ds.readable.getReader();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
        }
        const total = chunks.reduce((n, c) => n + c.length, 0);
        const out = new Uint8Array(total);
        let off = 0;
        for (const c of chunks) { out.set(c, off); off += c.length; }
        return new TextDecoder().decode(out);
    }
    window.OBSERVATORY_DATA = JSON.parse(await _obsDecompress(window.__OBS_RAW__));
"""
    else:
        data_script = f'window.OBSERVATORY_DATA = {payload_json};'
        decompress_block = ""

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
        {data_script}
    </script>

    <script>
        (async function() {{
{decompress_block}
            const res = (window.OBSERVATORY_DATA || {{}}).resources || {{}};
            if (res.css && res.css.length > 0) {{
                const style = document.createElement('style');
                style.textContent = res.css.map(function(s) {{
                    try {{ return atob(s); }} catch(_) {{ return s; }}
                }}).join('\\n');
                document.head.appendChild(style);
            }}
            if (res.js && res.js.length > 0) {{
                const script = document.createElement('script');
                script.textContent = res.js.map(function(s) {{
                    try {{ return atob(s); }} catch(_) {{ return s; }}
                }}).join(';\\n');
                document.body.appendChild(script);
            }}

{js_bundle}
        }})();
    </script>
</body>
</html>"""
