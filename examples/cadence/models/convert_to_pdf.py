#!/usr/bin/env python3
import markdown
from weasyprint import HTML

# Read markdown file
with open('/home/sraut/ext_main/cad_rlc/executorch/examples/cadence/models/vit_encoder_ops.md', 'r') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Wrap in HTML with styling
full_html = '''
<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
h1 { color: #333; border-bottom: 2px solid #333; }
h2 { color: #444; border-bottom: 1px solid #ccc; margin-top: 30px; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #f4f4f4; }
tr:nth-child(even) { background-color: #f9f9f9; }
code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
</style>
</head>
<body>
''' + html_content + '''
</body>
</html>
'''

# Generate PDF
output_path = '/home/sraut/ext_main/cad_rlc/executorch/examples/cadence/models/vit_encoder_ops.pdf'
HTML(string=full_html).write_pdf(output_path)
print(f'PDF created: {output_path}')
