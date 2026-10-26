"""Append compiled large static methods while preserving original PTE methods.

This prepares an experimental artifact, not a numerical or performance pass.
It does not export or requantize weights. GPU validation remains required.
Run with the ExecuTorch source/schema version used for the input artifacts.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    if dataclasses.is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, bytes):
        return dict(bytes=len(value), SHA256=sha(value))
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def fingerprint(value):
    return sha(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode())


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source', type=Path, required=True)
    a.add_argument('--baseline-pte', type=Path, required=True)
    a.add_argument('--candidate-pte', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.source.resolve().parent))
    from executorch.exir._serialize._program import deserialize_pte_binary, serialize_pte_binary
    from executorch.exir._serialize._named_data_store import NamedDataStore
    from executorch.exir.schema import DataLocation

    assert args.output.resolve() not in [args.baseline_pte.resolve(), args.candidate_pte.resolve()]
    args.output.mkdir(parents=True, exist_ok=False)
    raw_old, raw_new = args.baseline_pte.read_bytes(), args.candidate_pte.read_bytes()
    old, new = deserialize_pte_binary(raw_old), deserialize_pte_binary(raw_new)
    assert old.program.version == new.program.version
    assert old.mutable_data is None and new.mutable_data is None
    assert old.named_data is not None and new.named_data is not None
    assert not old.named_data.external_data and not new.named_data.external_data
    assert old.program.constant_buffer == new.program.constant_buffer, 'Global constant references differ'
    old_methods = {p.name: p for p in old.program.execution_plan}
    new_methods = {p.name: p for p in new.program.execution_plan}
    assert len(old_methods) == len(old.program.execution_plan)
    assert len(new_methods) == len(new.program.execution_plan)
    additions = ['forward_128', 'forward_256', 'forward_512']
    assert set(new_methods) == set(old_methods) | set(additions)
    assert not set(additions) & set(old_methods)
    assert 'forward_72' not in new_methods and 'forward_520' not in new_methods
    for name, plan in old_methods.items():
        if not plan.delegates:
            assert plan == new_methods[name], f'Metadata differs: {name}'
    def named_entries(pte):
        return {key: dict(SHA256=sha(bytes(pte.named_data.buffers[e.buffer_index])),
                          bytes=len(pte.named_data.buffers[e.buffer_index]))
                for key, e in pte.named_data.pte_data.items()}
    def method_identity(pte, plan):
        normalized = copy.deepcopy(plan)
        payloads = []
        for delegate in normalized.delegates:
            assert delegate.processed.location == DataLocation.INLINE
            payload = pte.program.backend_delegate_data[delegate.processed.index].data
            payloads.append(dict(bytes=len(payload), SHA256=sha(payload)))
            delegate.processed.index = 0
        return dict(plan_SHA256=fingerprint(normalized), delegate_payloads=payloads)
    def identities(pte):
        return {p.name: method_identity(pte, p) for p in pte.program.execution_plan}
    old_identities = identities(old)
    expected = dict(old_identities)
    expected.update({name: method_identity(new, new_methods[name]) for name in additions})
    merged = copy.deepcopy(old)
    for name in additions:
        plan = copy.deepcopy(new_methods[name])
        assert len(plan.inputs) == 3 and len(plan.outputs) == 1
        assert len(plan.delegates) == 1 and plan.delegates[0].id == 'CudaBackend'
        for delegate in plan.delegates:
            assert delegate.processed.location == DataLocation.INLINE
            payload = copy.deepcopy(new.program.backend_delegate_data[delegate.processed.index])
            delegate.processed.index = len(merged.program.backend_delegate_data)
            merged.program.backend_delegate_data.append(payload)
        merged.program.execution_plan.append(plan)
    named = NamedDataStore()
    named.merge_named_data_store(old.named_data)
    named.merge_named_data_store(new.named_data)
    merged.named_data = named.get_named_data_store_output()
    # Both original plans and their referenced compiled payloads are retained.
    assert merged.program.execution_plan[:len(old.program.execution_plan)] == old.program.execution_plan
    assert identities(merged) == expected
    def write_and_verify(pte, filename, expected_identities):
        path = args.output / filename
        with path.open('xb') as output:
            serialize_pte_binary(pte, extract_delegate_segments=True).write_to_file(output)
        restored = deserialize_pte_binary(path.read_bytes())
        assert identities(restored) == expected_identities
        assert restored.program.constant_buffer == old.program.constant_buffer
        assert restored.mutable_data is None
        assert named_entries(restored) == named_entries(pte)
        return dict(path=str(path.resolve()), bytes=path.stat().st_size, SHA256=sha(path.read_bytes()))
    # Preserve a serialization-only control for the required GPU comparison.
    control = write_and_verify(old, 'baseline-roundtrip.pte', old_identities)
    candidate = write_and_verify(merged, 'model.pte', expected)
    record = dict(status='CPU-assembly-verified-GPU-validation-required',
                  baseline=dict(path=str(args.baseline_pte.resolve()), SHA256=sha(raw_old)),
                  donor=dict(path=str(args.candidate_pte.resolve()), SHA256=sha(raw_new)),
                  source=str(args.source.resolve()), producer_SHA256=sha(Path(__file__).read_bytes()),
                  added_methods=additions, original_methods_preserved=old_identities,
                  new_methods={name: expected[name] for name in additions},
                  constant_buffers=[dict(bytes=len(v.storage), SHA256=sha(v.storage))
                                    for v in old.program.constant_buffer],
                  all_named_entries=named_entries(merged),
                  original_named_entries=named_entries(old),
                  unused_donor_named_blobs_retained=True,
                  baseline_roundtrip=control, candidate=candidate,
                  weights_requantized=False, weights_modified=False,
                  numerical_qualified=False, performance_qualified=False,
                  main_candidate_accepted=False)
    (args.output / 'assembly-review.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
