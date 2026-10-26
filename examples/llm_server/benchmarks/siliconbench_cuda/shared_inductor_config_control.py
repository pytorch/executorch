"""Export-process control for reproducible generated-kernel configurations.

Only choose configurations already supplied by Inductor for that kernel.
Exact JIT sources share a winner across exports. The full-KV write/reduction
control additionally matches its name after removing addressing-only fusion
labels, geometry hints and floating pointer types. This is an explicit
diagnostic: generated source, actual configurations, numerics and latency
must still be reviewed before any deployment. No compiler installation or
runtime source file is modified.
"""
import hashlib
import inspect
import json
from pathlib import Path
import re


def install_shared_inductor_config_control(output):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    output = Path(output)
    assert not output.exists()
    original_init = CachingAutotuner.__init__
    original_precompile = CachingAutotuner.precompile
    original_tune = CachingAutotuner.autotune_to_one_config
    signature = inspect.signature(original_init)
    winners, events = {}, []

    def config_record(config):
        return dict(kwargs=dict(config.kwargs), num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    num_ctas=getattr(config, 'num_ctas', 1),
                    maxnreg=getattr(config, 'maxnreg', None))

    def config_id(config):
        return json.dumps(config_record(config), sort_keys=True)

    def key_for(tuner):
        if tuner.custom_kernel:
            return None, False
        source = tuner.fn.src
        name = tuner.fn.__name__
        addressing_control = (
            name.startswith('triton_per_fused__to_copy_add_index_copy_mean_mul_pow_')
            and 'rsqrt' in name and 'slice_transpose' in name)
        common = dict(size_hints=tuner.size_hints,
                      heuristic=str(tuner.heuristic_type),
                      constants=tuner.triton_meta.get('constants', {}),
                      compute_capability=tuner.device_props.cc,
                      float_fusion=tuner.triton_meta.get('enable_fp_fusion'))
        if addressing_control:
            normalized = re.sub(r'_\d+$', '', name.replace('remainder_', '').replace('select_', ''))
            common['full_KV_write_reduction'] = normalized
            common['floating_signature'] = [value for value in tuner.triton_meta['signature'].values()
                                             if value in ('*bf16', '*fp16', '*fp32', '*fp64')]
        else:
            common['exact_JIT_source_SHA256'] = hashlib.sha256(source.encode()).hexdigest()
            common['signature'] = tuner.triton_meta['signature']
        return json.dumps(common, sort_keys=True, default=str), addressing_control

    def save():
        pending = output.with_suffix('.pending.json')
        pending.write_text(json.dumps(dict(
            exact_source_or_explicit_addressing_control=True,
            candidate_accepted=False, runtime_source_changed=False,
            winners=winners, events=events), indent=2) + '\n')
        pending.replace(output)

    def remember(tuner):
        key = getattr(tuner, '_siliconbench_config_key', None)
        if key is None or len(tuner.launchers) != 1:
            return
        winner = config_record(tuner.launchers[0].config)
        if key in winners:
            assert winners[key] == winner, 'Generated-kernel control lost its chosen configuration'
        else:
            winners[key] = winner
            events.append(dict(action='record', kernel=tuner.fn.__name__,
                addressing_control=tuner._siliconbench_addressing_control,
                config=winner))
        save()

    def init(tuner, *args, **kwargs):
        bound = signature.bind(tuner, *args, **kwargs)
        supplied_configs = bound.arguments['configs']
        original_init(tuner, *args, **kwargs)
        key, addressing = key_for(tuner)
        tuner._siliconbench_config_key = key
        tuner._siliconbench_addressing_control = addressing
        if key is None or key not in winners:
            return
        chosen = next((config for config in supplied_configs
                       if config_id(config) == json.dumps(winners[key], sort_keys=True)), None)
        assert chosen is not None, 'Recorded configuration is not a supplied candidate for this kernel'
        tuner.configs = [chosen]
        events.append(dict(action='reuse', kernel=tuner.fn.__name__,
            addressing_control=addressing, config=config_record(chosen),
            supplied_candidate_count=len(supplied_configs)))
        save()

    def precompile(tuner, *args, **kwargs):
        result = original_precompile(tuner, *args, **kwargs)
        remember(tuner)
        return result

    def tune(tuner, *args, **kwargs):
        result = original_tune(tuner, *args, **kwargs)
        remember(tuner)
        return result

    CachingAutotuner.__init__ = init
    CachingAutotuner.precompile = precompile
    CachingAutotuner.autotune_to_one_config = tune

    def restore():
        CachingAutotuner.__init__ = original_init
        CachingAutotuner.precompile = original_precompile
        CachingAutotuner.autotune_to_one_config = original_tune
        save()

    return restore
