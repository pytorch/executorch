"""Run stock server cells sequentially, retaining every failed attempt.

Only task-owned Supervisor configuration may be installed. This file changes
no runtime source, model artifact, existing result or existing service.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--native-inputs', type=Path)
    parser.add_argument('--correctness-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19082)
    parser.add_argument('--cpp-off-build', default='build-stock-native-evaluation')
    parser.add_argument('--variants', nargs='+', choices=['llamacpp-off', 'llamacpp-on', 'vllm', 'sglang', 'executorch'],
                        default=['llamacpp-off', 'vllm', 'sglang', 'llamacpp-on', 'executorch'])
    parser.add_argument('--cache-modes', nargs='+', choices=['off', 'on'], default=['off', 'on'])
    parser.add_argument('--concurrencies', nargs='+', type=int, choices=[1,2,4,8,16], default=[16,8,4,2,1])
    parser.add_argument('--install-supervisor', metavar='PROGRAM')
    parser.add_argument('--supervisor-config-dir', type=Path, default=Path('/etc/supervisor/conf.d'))
    args = parser.parse_args()
    assert not args.output.exists(), 'Retain existing attempts; inspect before using a new output directory'
    if args.install_supervisor:
        program = args.install_supervisor
        assert program and all(c.isalnum() or c in '-_' for c in program)
        config = args.supervisor_config_dir/(program+'.conf')
        assert not config.exists(), 'Never overwrite an existing Supervisor program'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        forwarded = list(sys.argv[1:])
        index = forwarded.index('--install-supervisor')
        del forwarded[index:index+2]
        command = [sys.executable, '-B', str(Path(__file__).resolve()), *forwarded]
        command_text = shlex.join(command).replace('%','%%')
        config.write_text(f'[program:{program}]\ncommand={command_text}\n'
                          'autostart=true\nautorestart=false\nstartsecs=2\nstopasgroup=true\nkillasgroup=true\n'
                          f'stdout_logfile={args.output.parent/(program+".log")}\n'
                          'stderr_redirect=true\nstdout_logfile_maxbytes=20MB\nstdout_logfile_backups=2\n')
        subprocess.run(['supervisorctl','reread'], check=True)
        subprocess.run(['supervisorctl','update',program], check=True)
        subprocess.run(['supervisorctl','status',program], check=True)
        return

    args.output.mkdir(parents=True)
    completed = []

    def state(status, **fields):
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                         utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                         completed=completed, **fields),indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    try:
        cases = [(cache,c,variant) for cache in args.cache_modes
                 for c in args.concurrencies for variant in args.variants]
        (args.output/'matrix.json').write_text(json.dumps(dict(model=args.model, cases=cases,
                                            acceptance_pending=True),indent=2)+'\n')
        for cache,c,variant in cases:
            name = f'agent-{variant}-prefix-{cache}-c{c}'
            state('running',current_case=name,total_cases=len(cases))
            backend = 'llamacpp' if variant.startswith('llamacpp-') else variant
            command = [args.python,'-B',str(Path(__file__).with_name('run_case.py')),
                       '--wrapper',str(args.wrapper),'--backend',backend,'--variant',variant,
                       '--model',args.model,'--prefix-cache',cache,'--client',str(args.client),
                       '--prompts',str(args.prompts),'--correctness-root',str(args.correctness_root),
                       '--python',args.python,'--concurrency',str(c),'--port',str(args.port),
                       '--output',str(args.output/name),'--lock',str(args.lock),
                       '--cpp-build',args.cpp_off_build if variant=='llamacpp-off' else 'build-stock-cublas']
            if args.native_inputs:
                command += ['--native-inputs',str(args.native_inputs)]
            with (args.output/(name+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result = json.loads((args.output/name/'progress.json').read_text())
            assert result['status']=='completed'
            completed.append(name)
        state('completed',total_cases=len(cases),performance_correctness_memory_collected=True,
              final_quality_cache_memory_and_best_ET_acceptance_pending=True)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
