"""Continue a known native job, export only minis, then collect four-arm data."""
import argparse, datetime, hashlib, json, os, subprocess, time, traceback
from pathlib import Path


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for n in ('native-preparation','gguf','output','lock','python'):a.add_argument('--'+n,type=Path,required=True)
    a.add_argument('--model',choices=['MG','Gemma'],required=True);args=a.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    package=Path(__file__).parent
    manifest=json.loads((package/'source-manifest.json').read_text())
    for name,h in {**manifest['sources'],**manifest['native_candidate_sources']}.items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==h,name
    def state(status,**extra):
        (args.output/'progress.json').write_text(json.dumps(dict(status=status,PID=os.getpid(),
            model=args.model,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_campaign_paused=True,mini_candidate_accepted=False,**extra),indent=2)+'\n')
    def run(script,argv):
        with (args.output/'controller.log').open('a') as log:
            subprocess.run([str(args.python),'-B','-u',str(package/script),*map(str,argv)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=21600)
    try:
        state('waiting-known-native-candidate')
        deadline=time.monotonic()+7200
        while True:
            p=json.loads((args.native_preparation/'progress.json').read_text())
            if p['status']=='native-mini-candidate-ready':break
            assert p['status']!='failed-native-mini-candidate',p
            assert (Path('/proc')/str(p['PID'])/'cmdline').exists(),p
            assert time.monotonic()<deadline,'native candidate timeout'
            time.sleep(20)
        state('exporting-same-GGUF-real-minis')
        exports=args.output/'exports'
        run('export_real_mini_pair.py',['--native-preparation',args.native_preparation,
            '--gguf',args.gguf,'--output',exports,'--lock',args.lock,'--model',args.model])
        state('measuring-four-arm-real-minis')
        run('run_real_mini_four_arms.py',['--native-preparation',args.native_preparation,
            '--export-job',exports,'--output',args.output/'four-arms','--lock',args.lock])
        state('real-mini-data-complete-review-required',result=str(args.output/'four-arms/result.json'))
    except BaseException as e:
        state('failed-real-mini-controller',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
