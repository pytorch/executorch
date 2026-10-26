"""Recompute actual full/smoke F1 and bind requests to the measured SDK.

Read original files only. This does not establish kernel/performance or main
acceptance. Partial progress is explicitly unqualified; complete review needs
all eight full 1146-item shots and twelve fixed balanced 64-item shots.
"""
import argparse, ast, collections, datetime, json, re
from pathlib import Path
from independent_session_full_native_binding import sha, verify_runtime

SCORER_SHA256 = '33c9ddaf9c7656a30b06d935d70a513e2dd737ca4f4e37f15ca746461a830c03'
CLIENT_SHA256 = 'd76fe1c8ba48ba54dfb397a381e7ea469205eae4cddad1538fb0f375c686e679'


def indexed(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {str(row['id']): row for row in rows}
    assert len(result) == len(rows), 'Duplicate input IDs'
    return result


def scorer_functions(path):
    assert sha(path) == SCORER_SHA256
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in
             ('_extract_json_category', 'extract_category', 'normalize_to_canonical') or
             isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and
             (target.id.startswith('METHOD_') or target.id.startswith('_') and
              target.id.endswith('_RE')) for target in node.targets)]
    namespace = {'json': json, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'verified extraction functions', 'exec'), namespace)
    return namespace


def final_object(content):
    text = (content or '').strip()
    marker = 'to=user<|message|>'
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    elif 'to=self' in text:
        text = ''
    for end in ('<|eot|>', '<|end|>', '<|eom|>'):
        text = text.removesuffix(end).strip()
    fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.IGNORECASE)
    try:
        value = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return None, False
    return value, bool(fenced)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('quality-job', 'public-inputs', 'native-job', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--full-model-profile', choices=('gemma4-48layers', 'muse-glimmer-52layers'), required=True)
    p.add_argument('--allow-completed-only', action='store_true')
    a = p.parse_args()
    assert not a.output.exists(), 'Preserve prior raw reviews'
    package = Path(__file__).resolve().parent
    verify_runtime(a.native_job, package)
    native_hash = sha(a.native_job/'native-gate.json')
    provenance = json.loads((a.quality_job/'provenance.json').read_text())
    assert provenance['native_runtime_gate_SHA256'] == native_hash
    assert provenance['full_profile'] == a.full_model_profile
    wrapper = a.quality_job/'serve-qualified-full.sh'
    assert provenance['wrapper_SHA256'] == sha(wrapper)
    correct = a.public_inputs/'correctness'
    functions = scorer_functions(correct/'scripts/score_f1.py')
    assert sha(correct/'scripts/run_eval.py') == CLIENT_SHA256
    categories_path = correct/'data/categories.json'
    categories = json.loads(categories_path.read_text())
    assert len(set(categories)) == len(categories)
    patterns = [(c, re.compile(r'\b'+re.escape(c.lower())+r'\b')) for c in sorted(categories, key=len, reverse=True)]
    canonical = {shot: {name: indexed(correct/f'prompts/{shot}shot/{name}.jsonl')
                 for name in ('prompts', 'labels')} for shot in (0, 5)}
    for rows in canonical.values():
        assert len(rows['labels']) == len(rows['prompts']) == 1146
        assert set(rows['labels']) == set(rows['prompts'])
    gemma = a.full_model_profile == 'gemma4-48layers'
    kwargs = {'enable_thinking': False} if gemma else {
        'enable_thinking': False, 'current_date': '2026-10-21', 'reasoning_strength': 'low'}
    evidence = {str(path.resolve()): sha(path) for path in (
        a.quality_job/'provenance.json', wrapper, a.native_job/'native-gate.json',
        correct/'scripts/score_f1.py', correct/'scripts/run_eval.py', categories_path)}
    reviews, missing, smoke_reference = [], [], None
    for prefix in ('off', 'on'):
        for c in (1, 2, 4, 8, 16):
            case = a.quality_job/f'correctness-agent-prefix-{prefix}-c{c}'
            scores_paths = [case/f'correctness/round-1/{shot}shot/scores.json' for shot in (0, 5)]
            if not any(path.is_file() for path in scores_paths):
                missing.extend((prefix, c, shot) for shot in (0, 5)); continue
            settings_path = case/'settings.json'
            settings = json.loads(settings_path.read_text())
            full = c in (1, 16); count = 1146 if full else 64
            assert settings['backend'] == 'executorch' and settings['split'] == 'agent'
            assert settings['model'] == ('gemma4-12b-it' if gemma else 'muse-glimmer-30b')
            assert settings['mode'] == 'correctness' and settings['concurrency'] == c
            assert settings['prefix_cache'] == (prefix == 'on')
            assert settings['quality_scope'] == ('full' if full else 'smoke')
            assert settings['correctness_rounds'] == 1 and settings['correctness_items'] == count
            assert settings['shots'] == [0, 5] and settings['wrapper_sha256'] == sha(wrapper)
            config = settings['correctness_request_configuration']
            assert config['chat_template_kwargs'] == kwargs
            assert config['temperature'] == 0 and config['top_p'] == 1
            assert config['scorer_sha256'] == SCORER_SHA256 and config['evaluation_client_sha256'] == CLIENT_SHA256
            assert config['max_tokens_override'] == (None if gemma else 1024)
            assert config['source_prompt_budgets_preserved'] == gemma
            bindings = [json.loads(line.split('FULL_MAIN_SERVER_BINDING:', 1)[1])
                        for line in (case/'server.log').read_text().splitlines() if line.startswith('FULL_MAIN_SERVER_BINDING:')]
            assert len(bindings) == 1
            binding = bindings[0]
            assert binding['native_runtime_gate_SHA256'] == native_hash
            assert binding['model_PTE_SHA256'] == provenance['artifact_SHA256']['model.pte']
            assert binding['prefix'] == prefix and binding['full_profile'] == a.full_model_profile
            assert binding['initial_KV'] == 512 and binding['logical_context'] == 131072
            assert binding['active_decode_limit'] == 16 and binding['static_forward_graphs_enabled']
            if not full:
                ids = sorted(settings['selected_smoke_ids'])
                assert len(ids) == len(set(ids)) == 64
                if smoke_reference is None: smoke_reference = ids
                assert ids == smoke_reference, 'Balanced sample must match at every concurrency/prefix'
            for shot, score_path in zip((0, 5), scores_paths):
                if not score_path.is_file(): missing.append((prefix, c, shot)); continue
                root = correct/f'prompts/{shot}shot' if full else case/f'smoke-inputs/{shot}shot'
                prompt_path, label_path = root/'prompts.jsonl', root/'labels.jsonl'
                prompts, labels = indexed(prompt_path), indexed(label_path)
                assert len(labels) == len(prompts) == count and set(labels) == set(prompts)
                for key in prompts:
                    assert prompts[key] == canonical[shot]['prompts'][key]
                    assert labels[key] == canonical[shot]['labels'][key]
                response_path = score_path.with_name('responses.jsonl')
                responses = indexed(response_path); assert set(responses) == set(labels)
                scores = json.loads(score_path.read_text())
                assert scores['provenance']['correctness_prompts_sha256'] == sha(prompt_path)
                pairs, errors, parsefails, strict, fences, final_disagreements, finishes = [], 0, 0, 0, 0, 0, collections.Counter()
                for key, row in responses.items():
                    request = row['request_parameters']
                    assert request['chat_template_kwargs'] == kwargs and request['temperature'] == 0 and request['top_p'] == 1
                    assert request['max_tokens'] == (prompts[key]['max_tokens'] if gemma else 1024)
                    assert request['stream'] is False
                    errors += bool(row.get('error')); finishes[row['finish_reason']] += 1
                    cat, _ = functions['extract_category']((row.get('content') or '')+(row.get('reasoning') or ''), patterns)
                    cat = functions['normalize_to_canonical'](cat, categories) if not row.get('error') else None
                    final_cat, _ = functions['extract_category'](row.get('content') or '', patterns)
                    final_cat = functions['normalize_to_canonical'](final_cat, categories) if not row.get('error') else None
                    final_disagreements += final_cat != cat
                    parsefails += cat is None
                    pairs.append((labels[key]['gold_category'], cat))
                    value, fenced = final_object(row.get('content'))
                    valid = isinstance(value, dict) and value.get('category') in categories
                    strict += valid and not fenced; fences += valid and fenced
                    if valid: assert value['category'] == cat
                gold = collections.Counter(g for g, pred in pairs); pred = collections.Counter(pred for g, pred in pairs)
                tp = collections.Counter(g for g, pred in pairs if g == pred)
                weighted = sum(gold[c]*2*tp[c]/(gold[c]+pred[c]) if gold[c]+pred[c] else 0 for c in categories)/count
                exact = sum(g == pred for g, pred in pairs)/count
                assert abs(weighted-scores['f1_weighted']) < 1e-10 and abs(exact-scores['exact_match']) < 1e-10
                assert errors == scores['request_errors'] and parsefails == scores['parse_failures']
                assert scores['n_scored'] == count and scores['n_missing_responses'] == scores['n_extra_responses'] == 0
                reviews.append(dict(prefix=prefix, concurrency=c, shots=shot, full=full, count=count,
                    independent_weighted_F1=weighted, independent_exact_match=exact, errors=errors,
                    parse_failures=parsefails, final_content_scoring_disagreements=final_disagreements, strict_complete_final_JSON=strict,
                    strict_complete_single_fenced_JSON=fences, finish_reasons=dict(finishes)))
                for path in (settings_path, case/'server.log', prompt_path, label_path, response_path, score_path):
                    evidence[str(path.resolve())] = sha(path)
    complete = not missing and len(reviews) == 20
    if not a.allow_completed_only: assert complete, missing
    raw_pass = complete and all(not row['errors'] and not row['parse_failures'] and
        not row['final_content_scoring_disagreements'] and row['finish_reasons'] == {'stop': row['count']} for row in reviews)
    result = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_profile=a.full_model_profile, native_runtime_gate_SHA256=native_hash,
        artifact_SHA256=provenance['artifact_SHA256'], reviewed_shots=len(reviews), expected_shots=20,
        expected_response_count=9936, actual_response_count=sum(row['count'] for row in reviews),
        complete=complete, missing=missing, full_task_F1_independently_reviewed=complete,
        strict_final_JSON_is_a_format_diagnostic_not_the_F1_scoring_contract=True,
        all_final_outputs_are_strict_single_JSON=complete and all(row['strict_complete_final_JSON']+row['strict_complete_single_fenced_JSON']==row['count'] for row in reviews),
        full_task_raw_quality_pass=raw_pass, shots=reviews, evidence_files_SHA256=evidence,
        final_JSON_format_scope='Complete bare final JSON or exactly one complete JSON fence; no partial or analysis JSON substituted',
        raw_input_request_and_runtime_binding_checked=True, original_files_modified=False,
        accuracy_is_not_universal_task_correctness=True, performance_qualified=False, main_candidate_accepted=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('full_profile', 'reviewed_shots', 'actual_response_count', 'complete', 'full_task_raw_quality_pass')}))


if __name__ == '__main__': main()
