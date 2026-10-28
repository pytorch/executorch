"""Run a CPU-only encoder timing with the exact HTTP-rendered prompt."""
import argparse,datetime,hashlib,json,subprocess,traceback
from pathlib import Path


def main(args):
    job,base=args.job,args.base;rc=1
    try:
        previous=base/'four-opt-MG-mini-HTTP-matched-native-token-diagnostic-20261028-v1'
        assert json.loads((previous/'exit-receipt.json').read_text())['returncode']==0
        HTTP=base/'four-opt-MG-mini-real-HTTP-routing-prefix-OFF-c1-c64-20261028-v1'
        cfg=json.loads((HTTP/'effective-config.json').read_text())
        prompt=previous/'rendered-prompt.txt';commands=json.loads((previous/'build-commands.json').read_text())
        compile_cmd=commands[0][:];compile_cmd[compile_cmd.index('-c')+1]=str(job/'profile_native_prompt_tokenizer.cpp');compile_cmd[compile_cmd.index('-o')+1]=str(job/'profile-tokenizer.o')
        libraries=[x for x in commands[-1] if any(x.endswith('/'+name) for name in ['libtokenizers.a','libre2.a','libpcre2-8.a']) or Path(x).name.startswith('libabsl_')];assert len(libraries)>3
        link=['/usr/bin/c++',str(job/'profile-tokenizer.o'),*libraries,'-pthread','-o',str(job/'profile-tokenizer')]
        (job/'build-commands.json').write_text(json.dumps([compile_cmd,link],indent=2))
        (job/'bindings.json').write_text(json.dumps(dict(prompt=str(prompt),prompt_SHA256=hashlib.sha256(prompt.read_bytes()).hexdigest(),CPU_only=True,model_execution=False),indent=2))
        with (job/'build.log').open('x') as log:
            for command in [compile_cmd,link]:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[str(job/'profile-tokenizer'),str(Path(cfg['tokenizer'])/'tokenizer.json'),str(prompt)]
        with (job/'stdout.json').open('x') as out,(job/'stderr.log').open('x') as err:subprocess.run(command,stdout=out,stderr=err,timeout=120,check=True)
        rc=0
    except BaseException:(job/'failure.txt').write_text(traceback.format_exc())
    finally:(job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,CPU_only=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',type=Path,required=True);main(p.parse_args())
