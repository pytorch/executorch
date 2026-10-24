"""Set the per-request context limit separately from initial KV and shared pool.

Use public launch switches only. A llama.cpp unified pool is shared by every
slot; its total size is distinct from the maximum context allowed in one slot.
This does not change the model's native position limit or make an old ET export
support a larger context. Audit a compatible artifact before launching ET.
"""
import argparse
from pathlib import Path
import re
import subprocess


def configure(body, maximum_context, initial_kv_capacity, cpp_pool_tokens=None):
    assert maximum_context in (12288, 131072)
    assert 0 < initial_kv_capacity <= maximum_context
    context = re.compile(r'^context=12288$', re.MULTILINE)
    assert len(context.findall(body)) == 1, 'Unexpected original context declaration'
    body = context.sub(f'context={maximum_context}', body)
    initial = re.compile(r'--initial-kv-capacity [0-9]+')
    assert len(initial.findall(body)) == 1, 'Unexpected initial KV argument'
    body = initial.sub(f'--initial-kv-capacity {initial_kv_capacity}', body)
    pool = '--ctx-size "$((context*active))"'
    assert body.count(pool) == 1, 'Expected the reviewed unified llama.cpp wrapper'
    assert body.count('--kv-unified-per-slot "$context"') == 1
    assert body.count('--kv-unified') == 2
    if cpp_pool_tokens is not None:
        assert cpp_pool_tokens >= maximum_context
        body = body.replace(pool, f'--ctx-size {cpp_pool_tokens}')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-context', type=int, choices=[12288,131072], required=True)
    parser.add_argument('--initial-kv-capacity', type=int, default=512)
    parser.add_argument('--cpp-pool-tokens', type=int)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing launchers'
    body = configure(args.base.read_text(), args.max_context,
                     args.initial_kv_capacity, args.cpp_pool_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body)
    subprocess.run(['bash', '-n', str(args.output)], check=True)


if __name__ == '__main__':
    main()
