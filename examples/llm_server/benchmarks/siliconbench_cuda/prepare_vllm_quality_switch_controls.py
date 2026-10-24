"""Clone a stock vLLM launch wrapper for two official correctness controls.

The original wrapper, framework source, weights and results stay unchanged.
These controls change only the supplied model-runner or eager-execution switch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--wrapper-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.wrapper) == args.wrapper_sha256
    assert not args.output.exists(), 'Preserve existing controls and records'
    original = args.wrapper.read_text()
    runner = 'export VLLM_USE_V2_MODEL_RUNNER=1'
    assert original.count(runner) == 1
    pattern = re.compile(r"(?m)^([ \t]*)--compilation-config '([^'\n]+)'([ \t]+\\)?[ \t]*$")
    matches = list(pattern.finditer(original))
    assert len(matches) == 1
    graph = json.loads(matches[0].group(2))
    assert graph['cudagraph_mode'] == 'FULL_AND_PIECEWISE'
    assert graph['cudagraph_capture_sizes'] == [1,2,4,8,16]
    versions = {
        'v1': original.replace(runner, 'export VLLM_USE_V2_MODEL_RUNNER=0'),
        'v2-eager': pattern.sub(lambda m: m.group(1)+'--enforce-eager'+(m.group(3) or ''), original),
    }
    args.output.mkdir(parents=True)
    records = []
    for name, text in versions.items():
        path = args.output/('serve-'+name+'.sh')
        path.write_text(text)
        path.chmod(0o700)
        subprocess.run(['bash', '-n', str(path)], check=True, timeout=30)
        records.append(dict(control=name, wrapper=str(path.resolve()), sha256=digest(path),
                            model_quantization_precision_KV_and_payload_unchanged=True))
    assert digest(args.wrapper) == args.wrapper_sha256
    record = dict(original_wrapper=str(args.wrapper.resolve()), original_sha256=args.wrapper_sha256,
                  controls=records, official_switches_only=True, no_framework_or_model_edit=True,
                  no_model_loaded=True, no_correctness_or_performance_acceptance=True)
    (args.output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
