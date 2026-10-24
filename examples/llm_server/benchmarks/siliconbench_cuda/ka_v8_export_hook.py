"""Prepared optional ET export hook; complete kernel gates required first."""
from pathlib import Path
import hashlib
import json


def _configure_export(native_family, *, capability, accepted_report,
                      candidate_source, a100_control_sha256, job_dir, validated_routes):
    from ka_v8_family_adapter import prepare_family
    import torch
    from candidates_w4a16_v8.aoti import config_index
    assert capability == (12, 0)
    assert len(a100_control_sha256) == 64 and all(c in '0123456789abcdef' for c in a100_control_sha256)
    routes, receipt = validated_routes(accepted_report, candidate_source)
    receipt['A100_unchanged_native_control_sha256'] = a100_control_sha256
    receipt['capability'] = list(capability)
    receipt['native_source_sha256'] = 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    family = prepare_family(native_family, capability=capability, accepted_routes=routes)
    original_export = torch.export.export
    log = Path(job_dir)/'actual-KA-export-routes.jsonl'

    def instrumented_export(module, *args, **kwargs):
        result = original_export(module, *args, **kwargs)
        nodes = []
        for node in result.graph_module.graph.nodes:
            if node.op != 'call_function' or 'kernelagent.w4a16_v8' not in str(node.target):
                continue
            x, q = node.args[:2]
            xv, qv = x.meta.get('val'), q.meta.get('val')
            assert xv is not None and qv is not None
            shape = (xv.shape[0], qv.shape[0], xv.shape[1])
            assert all(isinstance(v, int) for v in shape), 'No accepted candidate on an unproven dynamic interval'
            assert shape in routes and shape[0] in [8, 16]
            assert node.args[-1] == config_index(routes[shape])
            nodes.append({'op': str(node.target), 'shape': list(shape),
                          'config_index': node.args[-1], 'activation_dtype': str(xv.dtype)})
        first = args[0][0] if args and args[0] else None
        record = {'module_class': type(module).__name__,
                  'first_input_shape': [str(v) for v in first.shape] if hasattr(first, 'shape') else None,
                  'candidate_nodes': nodes, 'candidate_count': len(nodes),
                  'all_candidate_rows_are_8_or_16': True}
        with log.open('a') as output:
            output.write(json.dumps(record)+'\n')
        return result

    torch.export.export = instrumented_export
    receipt['hook_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return family, receipt


def audit_route_log(job_dir, receipt):
    rows = [json.loads(line) for line in (Path(job_dir)/'actual-KA-export-routes.jsonl').read_text().splitlines()]
    candidates = [n for row in rows for n in row['candidate_nodes']]
    assert candidates, 'No candidate reached the full exported model; keep baseline artifact'
    assert all(n['shape'][0] in [8, 16] and n['activation_dtype']=='torch.bfloat16' for n in candidates)
    receipt['FX_exports'] = rows
    receipt['actual_candidate_nodes'] = len(candidates)
    receipt['observed_candidate_shapes'] = sorted({tuple(n['shape']) for n in candidates})
    receipt['whole_server_perf_memory_F1_acceptance_pending'] = True
    return receipt


def configure_gemma_export(native_family, **kwargs):
    from ka_v8_route_gate import validated_routes
    return _configure_export(native_family,validated_routes=validated_routes,**kwargs)


def configure_mg_export(native_family, **kwargs):
    from ka_v8_mg_route_gate import validated_routes
    return _configure_export(native_family,validated_routes=validated_routes,**kwargs)
