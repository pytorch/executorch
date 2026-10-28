"""Preserve original boundary checks and extend paired decode coverage to c2/c4.

Run after strict numerical/lifecycle checks; preserve the original thresholds.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    p = args.job
    assert p.is_dir() and not (p/'boundary-config.json').exists()
    rc = 1
    try:
        handle = json.loads((args.qualification_job/'launch-receipt.json').read_text())
        proc = Path('/proc')/str(handle['PID'])
        while not (args.qualification_job/'exit-receipt.json').exists():
            assert proc.exists(), 'Prerequisite handle disappeared without terminal receipt'
            fields=(proc/'stat').read_text().split(') ',1)[1].split()
            assert int(fields[19]) == handle['start_ticks'] and fields[0] != 'Z'
            assert hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest() == handle['command_SHA256']
            time.sleep(15)
        assert json.loads((args.qualification_job/'exit-receipt.json').read_text())['returncode'] == 0
        previous = json.loads((args.qualification_job/'result.json').read_text())
        assert previous['original_numeric_gate_pass'] and previous['lifecycle_returncode'] == 0
        assert previous['export_control_gate_passed']
        assert not args.pipeline_output.exists()
        assert shutil.disk_usage(args.pipeline_output.parent).free >= 3*2**30
        cfg = json.loads((args.qualification_job/'effective-config.json').read_text())
        boundary = dict(cfg, expected_SHA256={k:digest(cfg[k])
            for k in ['probe','baseline_PTE','PTE','PTD']})
        (p/'boundary-config.json').write_text(json.dumps(boundary,indent=2))
        template = args.template
        for name in ['collector.py','run_double_ring_burst_qualification.py']:
            (p/name).write_bytes((template/name).read_bytes())
        pipeline_source = (p/'run_double_ring_burst_qualification.py').read_text()
        marker = "('decode1',1,0),('decode8',8,0)"
        assert pipeline_source.count(marker) == 1
        pipeline_source = pipeline_source.replace(marker,
            "('decode1',1,0),('decode2',2,0),('decode4',4,0),('decode8',8,0)")
        (p/'run_double_ring_burst_qualification.py').write_text(pipeline_source)
        (p/'source-bindings.json').write_text(json.dumps(dict(
            files={n:digest(p/n) for n in ['collector.py','run_double_ring_burst_qualification.py',
                'diagnose_ring_prefill_boundaries.py']},
            original_pipeline_SHA256=digest(template/'run_double_ring_burst_qualification.py'),
            only_pipeline_change='Add decode2 and decode4 cases',
            all_thresholds_unchanged=True,export_control_gate_passed=True),indent=2))
        env = dict(os.environ, LD_LIBRARY_PATH=cfg['native_lib'])
        commands = [[args.python,'-B','-u',str(p/'diagnose_ring_prefill_boundaries.py'),
            '--config',str(p/'boundary-config.json'),'--output',str(p/'boundaries')]]
        pair = p/'paired'
        pair.mkdir()
        for arm, pte in [('baseline',cfg['baseline_PTE']),('ring',cfg['PTE'])]:
            folder=pair/arm;folder.mkdir()
            (folder/'model.pte').symlink_to(pte)
            (folder/'aoti_cuda_blob.ptd').symlink_to(cfg['PTD'])
        pipeline=json.loads((template/'pipeline-config.json').read_text())
        pipeline.update(output=str(args.pipeline_output),collector=str(p/'collector.py'),
            baseline_artifact=str(pair/'baseline'),candidate_artifact=str(pair/'ring'))
        (p/'pipeline').symlink_to(args.pipeline_output)
        (p/'pipeline-config.json').write_text(json.dumps(pipeline,indent=2))
        commands.append([args.python,'-B','-u',str(p/'run_double_ring_burst_qualification.py'),
            '--config',str(p/'pipeline-config.json')])
        (p/'commands.json').write_text(json.dumps(commands,indent=2))
        returncodes=[]
        for name,cmd in zip(['boundaries','pipeline'],commands):
            (p/'progress.json').write_text(json.dumps(dict(status=name,mini_only=True,
                full_model_qualified=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
            with (p/(name+'.log')).open('x') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                returncodes.append(result.returncode)
        (p/'result.json').write_text(json.dumps(dict(returncodes=returncodes,mini_only=True,
            full_model_qualified=False,export_control_gate_passed=True,
            raw_independent_review_required=True),indent=2))
        rc = 0
    except BaseException:
        (p/'failure.txt').write_text(traceback.format_exc())
    finally:
        (p/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_model_qualified=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['job','qualification-job','template']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--pipeline-output',type=Path,required=True)
    main(parser.parse_args())
