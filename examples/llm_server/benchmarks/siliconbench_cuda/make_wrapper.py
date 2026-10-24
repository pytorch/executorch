"""Expose cache and llama.cpp build selection in an existing benchmark wrapper.

Only launch arguments change; framework source, templates and model paths are
preserved. Review the generated script and validate it before serving.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists(), 'Use a new path; preserve previous launchers'
body = args.base.read_text()
markers = {
    '--no-cache-prompt --cache-ram 0 --cache-reuse 0': '"${cpp_cache_args[@]}"',
    '--prefix-cache-entries 0': '--prefix-cache-entries "$et_prefix_entries"',
    '--disable-radix-cache': '"${sg_cache_args[@]}"',
    '--no-enable-prefix-caching': '"${vl_cache_args[@]}"',
    '/build-stock-cublas/bin/llama-server': '/${SB_CPP_BUILD}/bin/llama-server',
}
for before, after in markers.items():
    assert body.count(before) == 1, f'Unexpected launch wrapper: {before}'
    body = body.replace(before, after)
assert body.count('case "$backend" in') == 1
switches = '''
SB_CPP_BUILD=${SB_CPP_BUILD:-build-stock-cublas}
case "${SB_PREFIX_CACHE:?Set SB_PREFIX_CACHE to on or off}" in
  on)
    cpp_cache_args=(--cache-prompt --cache-ram "${SB_CPP_CACHE_RAM_MIB:-8192}" --cache-reuse 0)
    et_prefix_entries=${SB_ET_PREFIX_ENTRIES:-16}
    sg_cache_args=()
    vl_cache_args=(--enable-prefix-caching)
    ;;
  off)
    cpp_cache_args=(--no-cache-prompt --cache-ram 0 --cache-reuse 0)
    et_prefix_entries=0
    sg_cache_args=(--disable-radix-cache)
    vl_cache_args=(--no-enable-prefix-caching)
    ;;
  *) exit 2 ;;
esac
'''
body = body.replace('case "$backend" in', switches+'\ncase "$backend" in')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(body)
subprocess.run(['bash', '-n', str(args.output)], check=True)
