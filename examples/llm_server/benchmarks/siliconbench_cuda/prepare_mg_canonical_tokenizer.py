"""Download pinned public MG tokenizer assets identical to the main runs.

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

REPOSITORY='meta-models/Muse-Glimmer-30B'
REVISION='cec94fcfd6512f2cfe708138c32533b346ea4f69'
FILES={'tokenizer.json': ('c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d', 28129897), 'tokenizer_config.json': ('781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5', 79936), 'chat_template.jinja': ('cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678', 9992), 'config.json': ('5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14', 5109), 'generation_config.json': ('1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35', 202)}


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
        state('canonical-MG-tokenizer-ready',tokenizer=str(tokenizer))
    except Exception as error:
        state('failed',detail=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
