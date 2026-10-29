"""Compare actual cache reuse with exact-token online history references."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys


def common_prefix(a,b):
    limit=min(len(a),len(b))
    i=0
    while i<limit and a[i]==b[i]:i+=1
    return i


def main(job):
    cfg=json.loads((job/'diagnostic-config.json').read_text())['main_configuration']
    sys.path.insert(0,cfg['HTTP_source'])
    from executorch.examples.llm_server.python.chat_template import ChatTemplate
    from executorch.examples.llm_server.python.protocol import ChatMessage
    template=ChatTemplate(cfg['tokenizer'],assistant_header='<|turn>model\n',strip_rendered_bos=False,
                          append_generation_prompt_after_tool_response=False)
    template.generation_preamble()
    tokenized={};input_proof={}
    for split in ['chat','agent']:
        promptfile=job/(split+'_benchmark_prompts.json')
        value=json.loads(promptfile.read_text())
        prompts=value['prompts'] if isinstance(value,dict) else value
        pool={}
        for prompt in prompts:
            rendered=template.render([ChatMessage.model_validate(m) for m in prompt['messages']],
                tools=None,template_kwargs=dict(enable_thinking=False,current_date='2026-10-21',reasoning_strength='high'))
            tokens=template._hf.encode(rendered,add_special_tokens=False)
            assert len(tokens)==template.count_tokens(rendered)
            pool[prompt['name']]=tokens
        assert len(pool)==len(prompts)
        tokenized[split]=dict(pool=pool,primer=[p['name'] for p in prompts[:3]])
        input_proof[split]=dict(source_SHA256=hashlib.sha256(promptfile.read_bytes()).hexdigest(),
            unique_prompts=len(pool),unique_token_sequences=len({tuple(tokens) for tokens in pool.values()}),
            token_counts={name:len(tokens) for name,tokens in pool.items()})
    results=[]
    for case in sorted((job/'cases').iterdir()):
        if not case.is_dir():continue
        settings=json.loads((case/'settings.json').read_text())
        split=settings['split'];pool=tokenized[split]['pool'];all_available=[];four_lru=[];all_submitted=[]
        for round_id in range(1,5):
            path=case/f'repeat-{round_id}/outputs.jsonl'
            if not path.is_file():continue
            rows=[json.loads(line) for line in path.open() if line.strip()]
            assert len(rows)==100 and len({row['request_idx'] for row in rows})==100
            for name in tokenized[split]['primer']:
                tokens=pool[name];all_available.append(tokens);all_submitted.append(tokens)
                key=tuple(tokens)
                if key in four_lru:four_lru.remove(key)
                four_lru.insert(0,key);four_lru=four_lru[:4]
            events=[]
            for row in rows:
                events += [(row['request_started_monotonic_s'],1,row),(row['request_finished_monotonic_s'],0,row)]
            actual_total=0;demand_total=0;available_total=0;submission_total=0;lru_total=0;detail=[]
            for _,kind,row in sorted(events,key=lambda item:(item[0],item[1],item[2]['request_idx'])):
                tokens=pool[row['prompt_name']]
                if kind==0:
                    all_available.append(tokens)
                    key=tuple(tokens)
                    if key in four_lru:four_lru.remove(key)
                    four_lru.insert(0,key);four_lru=four_lru[:4]
                    continue
                assert len(tokens)==row['prompt_tokens'],(case.name,row['request_idx'],len(tokens),row['prompt_tokens'])
                cap=max(0,len(tokens)-1)
                available=min(cap,max((common_prefix(tokens,old) for old in all_available),default=0))
                submitted=min(cap,max((common_prefix(tokens,old) for old in all_submitted),default=0))
                lru=min(cap,max((common_prefix(tokens,old) for old in four_lru),default=0))
                assert row['cached_prompt_tokens'] is not None,'Require explicit server cached-token usage'
                actual=row['cached_prompt_tokens'];actual_total+=actual;demand_total+=len(tokens)
                available_total+=available;submission_total+=submitted;lru_total+=lru
                detail.append(dict(request_idx=row['request_idx'],prompt_name=row['prompt_name'],prompt_tokens=len(tokens),
                    actual_cached_tokens=actual,unlimited_completed_history_prefix=available,
                    unlimited_submitted_history_optimistic_upper_bound=submitted,simplified_four_entry_whole_prompt_LRU=lru))
                all_submitted.append(tokens)
            results.append(dict(case=case.name,round=round_id,measured=round_id>=2,requests=100,input_tokens=demand_total,
                actual_cached_tokens=actual_total,actual_token_hit_rate=actual_total/demand_total,
                theoretical_unlimited_completed_cached_tokens=available_total,theoretical_unlimited_completed_token_hit_rate=available_total/demand_total,
                theoretical_unlimited_submitted_upper_bound_tokens=submission_total,
                simplified_four_entry_whole_prompt_LRU_tokens=lru_total,
                actual_request_hit_rate=sum(d['actual_cached_tokens']>0 for d in detail)/100,
                theoretical_completed_request_hit_rate=sum(d['unlimited_completed_history_prefix']>0 for d in detail)/100,
                per_request=sorted(detail,key=lambda r:r['request_idx'])))
    document=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exact_template_and_token_usage_verified=True,
        final_token_left_for_forward=True,input_proof=input_proof,results=results,
        unlimited_completed_reference='All completed prompt prefixes plus the three sequential warmups, including previous rounds on the same server. Inflight checkpoint capture may provide reuse sooner; this reference alone does not prove implementation misses.',
        unlimited_submitted_reference='Optimistic upper bound; earlier submitted but unfinished requests may not yet have reusable KV.',
        four_entry_reference='Simplified whole-prompt LRU after completion, deduplicated by the complete token ID sequence rather than prompt name; does not model periodic checkpoints, capture-lane skips or clone failure. Exact currently retained-candidate theory must be measured by the diagnostic native trace.',
        no_framework_or_kernel_changed=True)
    (job/'theoretical-hit-review-token-dedup-v2.json').write_text(json.dumps(document,indent=2))
    print(json.dumps(dict(reviewed_rounds=len(results),usage_token_counts_match=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    main(parser.parse_args().job)
