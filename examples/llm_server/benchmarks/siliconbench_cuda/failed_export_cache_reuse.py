"""Reuse only compilation caches from a confirmed terminal owned export.

Preserve previous artifacts, logs and source. Re-export into a new sibling job;
cache reuse does not qualify model numerics or performance.
"""
import hashlib
import json
from pathlib import Path
import shutil


def verify_failed_export(previous, output, script_names):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    assert previous != output and previous.parent == output.parent
    progress = json.loads((previous/'progress.json').read_text())
    assert progress['status'] == 'failed', 'Never restart a healthy export'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command=(proc/'cmdline').read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(previous).encode() in command and any(name.encode() in command for name in script_names):
            assert b'--reuse-failed-job' in command and str(output).encode() in command, \
                'Previous export process is still live'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(16*1024*1024):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    records={}
    # Focused traversal of an explicitly selected owned cache only.
    for path in root.rglob('*'):
        relative=str(path.relative_to(root))
        assert not path.is_symlink(), 'Do not copy cache links outside the owned job'
        if path.is_file():
            records[relative]=dict(bytes=path.stat().st_size,SHA256=digest(path))
    return records


def copy_compiler_caches(previous_cache, new_cache, receipt):
    previous_cache, new_cache, receipt=map(Path,(previous_cache,new_cache,receipt))
    assert not receipt.exists()
    copied={}
    for name in ('inductor','triton'):
        source, target=previous_cache/name,new_cache/name
        if not source.exists():
            continue
        assert not target.exists()
        before=inventory(source)
        shutil.copytree(source,target)
        assert inventory(source)==before and inventory(target)==before
        copied[name]=before
    receipt.write_text(json.dumps(dict(previous_cache=str(previous_cache),
        copied_cache_files=copied,previous_cache_unchanged=True,
        previous_artifacts_and_logs_preserved=True,model_acceptance_pending=True),indent=2)+'\n')
