"""Download pinned public Gemma tokenizer assets identical to the main runs.

Only create a new owned directory. No model weights are downloaded; existing
tokenizers, model configuration, servers and experiment records are untouched.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import traceback
import urllib.request

REPOSITORY='google/gemma-4-12B-it'
REVISION='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
FILES={
    'config.json':('478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9',4423),
    'tokenizer.json':('cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',32169626),
    'tokenizer_config.json':('a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9',3089),
    'chat_template.jinja':('ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',18683),
    'generation_config.json':('a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e',260),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    def state(status,**fields):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,
            candidate_accepted=False,**fields),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        tokenizer=args.output/'tokenizer';tokenizer.mkdir()
        records={}
        for name,(expected,count) in FILES.items():
            URL=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
            state('downloading-pinned-public-tokenizer',file=name)
            with urllib.request.urlopen(URL,timeout=120) as response:
                raw=response.read()
            assert len(raw)==count and hashlib.sha256(raw).hexdigest()==expected,name
            (tokenizer/name).write_bytes(raw)
            records[name]=dict(URL=URL,SHA256=expected,bytes=count)
        (args.output/'tokenizer-source.json').write_text(json.dumps(dict(
            public_repository=REPOSITORY,revision=REVISION,files=records,
            tokenizer=str(tokenizer),matches_main_canonical_asset_bytes=True,
            no_model_weights_downloaded=True,existing_records_unchanged=True),indent=2)+'\n')
        state('canonical-Gemma-tokenizer-ready',tokenizer=str(tokenizer))
    except Exception as error:
        state('failed',detail=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
