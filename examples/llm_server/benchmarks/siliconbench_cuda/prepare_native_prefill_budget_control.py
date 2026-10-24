"""Clone a benchmark launch wrapper with official native prefill budget flags.

The original launch wrapper, runtime source and model are left untouched.
The clone changes exactly the three vLLM/SGLang budget arguments below.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--wrapper-sha256', required=True)
    parser.add_argument('--original-budget', type=int, required=True)
    parser.add_argument('--matched-budget', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.original_budget > args.matched_budget > 0
    assert not args.output.exists(), 'Preserve all previous attempts'
    raw = args.wrapper.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == args.wrapper_sha256, 'Source wrapper changed'
    text = raw.decode()
    replacements = []
    for flag in ['--max-num-batched-tokens', '--chunked-prefill-size', '--max-prefill-tokens']:
        old = f'{flag} {args.original_budget}'
        new = f'{flag} {args.matched_budget}'
        assert text.count(old) == 1, f'Expected exactly one original argument: {old}'
        text = text.replace(old, new, 1)
        replacements.append(dict(flag=flag, original=args.original_budget, matched=args.matched_budget))
    args.output.mkdir(parents=True)
    clone = args.output/'serve-native-prefill-control.sh'
    clone.write_text(text)
    clone.chmod(0o755)
    subprocess.run(['bash', '-n', str(clone)], check=True)
    assert args.wrapper.read_bytes() == raw, 'Original wrapper must remain unchanged'
    receipt = dict(original_wrapper=str(args.wrapper), original_sha256=digest,
                   cloned_wrapper=str(clone), cloned_sha256=hashlib.sha256(clone.read_bytes()).hexdigest(),
                   replacements=replacements, official_cli_arguments_only=True,
                   runtime_source_changed=False, model_changed=False,
                   comparison_scope='Supplemental matched-prefill-budget control; primary results retained')
    (args.output/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
