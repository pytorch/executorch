"""Compare prepared native decoding with the actual HTTP prompt/tokenizer."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def main(args):
    job,base=args.job,args.base
    assert job.is_dir() and not (job/'routing_mini.cpp').exists()
    rc=1
    try:
        HTTP=base/'four-opt-MG-mini-real-HTTP-routing-prefix-OFF-c1-c64-20261028-v1'
        assert json.loads((HTTP/'exit-receipt.json').read_text())['returncode']==0
        cfg=json.loads((HTTP/'effective-config.json').read_text())
        env=dict(os.environ,PYTHONPATH=cfg['HTTP_source'])
        render="""import json,sys
from pathlib import Path
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.models.muse_glimmer.serving import serve
cfg=json.loads(Path(sys.argv[1]).read_text());req=json.loads(Path(sys.argv[2]).read_text())
t=ChatTemplate(cfg['tokenizer'],assistant_header=serve._ASSISTANT_HEADER,strip_rendered_bos=True,append_generation_prompt_after_tool_response=True)
p=t.render([ChatMessage.model_validate(m) for m in req['messages']],tools=None,template_kwargs={})
Path(sys.argv[3]).write_text(p)
"""
        prompt=job/'rendered-prompt.txt'
        subprocess.run([cfg['HTTP_python'],'-B','-c',render,str(HTTP/'effective-config.json'),str(HTTP/'request.json'),str(prompt)],env=env,check=True)
        previous=base/'four-opt-MG-mini-routing-prefix-OFF-c1-c64-20261028-v2'
        source=(previous/'routing_mini.cpp').read_text()
        source=source.replace('#include <pytorch/tokenizers/tokenizer.h>','#include <pytorch/tokenizers/tokenizer.h>\n#include <pytorch/tokenizers/hf_tokenizer.h>\n#include <fstream>')
        source=source.replace('DEFINE_string(pte,', 'DEFINE_string(rendered_prompt_file,"","Exact server-rendered prompt");\nDEFINE_string(tokenizer_file,"","Canonical tokenizer");\nDEFINE_bool(real_text_decode,false,"Use real tokenizer for response events");\nDEFINE_string(pte,',1)
        source=source.replace('struct Tokenizer final:tokenizers::Tokenizer {','struct Tokenizer final:tokenizers::Tokenizer {\n  tokenizers::HFTokenizer actual;\n  Tokenizer(){if(FLAGS_real_text_decode)require(actual.load(FLAGS_tokenizer_file)==tokenizers::Error::Ok,"decoder load");}',1)
        marker='tokenizers::Result<std::string> decode(uint64_t,uint64_t token,bool) const override{return std::string("t")+std::to_string(token);}'
        assert source.count(marker)==1
        source=source.replace(marker,'tokenizers::Result<std::string> decode(uint64_t previous,uint64_t token,bool skip) const override{if(FLAGS_real_text_decode)return actual.decode(previous,token,skip);return std::string("t")+std::to_string(token);}')
        marker='    std::vector<std::vector<uint64_t>> histories(FLAGS_concurrency);'
        assert source.count(marker)==1
        source=source.replace(marker,"""    tokenizers::HFTokenizer exact_encoder;require(exact_encoder.load(FLAGS_tokenizer_file)==tokenizers::Error::Ok,"exact prompt encoder");
    std::ifstream rendered_stream(FLAGS_rendered_prompt_file);require(bool(rendered_stream),"rendered prompt file");
    std::string rendered((std::istreambuf_iterator<char>(rendered_stream)),std::istreambuf_iterator<char>());
    auto encoded=exact_encoder.encode(rendered,0,0);require(encoded.ok(),"encode exact rendered prompt");
    std::vector<uint64_t> fixed_input{200000};fixed_input.insert(fixed_input.end(),encoded->begin(),encoded->end());
    require(fixed_input.size()==static_cast<size_t>(FLAGS_prompt_tokens),"actual HTTP prompt length match");
    result["real_text_decode"]=FLAGS_real_text_decode;result["actual_CPP_tokenizer_prompt_ids"]=fixed_input;
"""+marker)
        marker='histories[owner].clear();for(int n=0;n<FLAGS_prompt_tokens;++n)histories[owner].push_back(3+(burst*FLAGS_prompt_tokens+n+owner*17)%18000);'
        assert source.count(marker)==1;source=source.replace(marker,'histories[owner]=fixed_input;')
        (job/'routing_mini.cpp').write_text(source)
        commands=json.loads((previous/'build-commands.json').read_text());original_compile=next(c for c in commands if any(a.endswith('/routing_mini.cpp') for a in c))
        compile_cmd=original_compile[:];compile_cmd[compile_cmd.index('-c')+1]=str(job/'routing_mini.cpp');compile_cmd[compile_cmd.index('-o')+1]=str(job/'routing-mini.o')
        link=commands[-1][:];link[link.index('-o')+1]=str(job/'routing-mini')
        index=link.index(str(previous/'routing-mini.o'));link[index]=str(job/'routing-mini.o')
        (job/'build-commands.json').write_text(json.dumps([compile_cmd,link],indent=2))
        (job/'bindings.json').write_text(json.dumps(dict(original_source_SHA256=hashlib.sha256((previous/'routing_mini.cpp').read_bytes()).hexdigest(),
            modified_source_SHA256=hashlib.sha256(source.encode()).hexdigest(),
            rendered_prompt_SHA256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
            actual_HTTP_bindings=json.loads((HTTP/'bindings.json').read_text()),
            only_changes='Exact HTTP-rendered input and optional real response decoder',mini_only=True,full_main_unchanged=True),indent=2))
        with (job/'build.log').open('x') as log:
            for command in [compile_cmd,link]:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        sdk=base/'MG-mini-required-side-binding-native-release-20261027-v2/native/executorch'
        env=dict(os.environ,LD_LIBRARY_PATH=str(sdk/'cmake-out/lib'),ET_CUDA_RING_JOURNAL_BUFFERS='0',ET_CUDA_RING_ARCHIVE_BATCH_ROWS='64',ET_CUDA_SHARED_IDLE_REFRESH='1')
        records=[]
        with (base/'gpu-benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for round_id in range(1,5):
                for c in [1,2,4,8,16,32,64]:
                    for real in [False,True]:
                        folder=job/'cases'/str(round_id)/str(c)/str(int(real));folder.mkdir(parents=True)
                        argv=[str(job/'routing-mini'),'--pte='+cfg['PTE'],'--data_path='+cfg['PTD'],
                            '--concurrency='+str(c),'--budget=4096','--prompt_tokens=1067','--prefix_entries=0',
                            '--cold_sessions=true','--admission_quiet_us=300','--admission_maximum_us=2000',
                            '--diagnostic_admission_barrier=false','--shared_idle_refresh=false',
                            '--rendered_prompt_file='+str(prompt),'--tokenizer_file='+str(Path(cfg['tokenizer'])/'tokenizer.json'),
                            '--real_text_decode='+str(real).lower()]
                        (job/'progress.json').write_text(json.dumps(dict(round=round_id,c=c,real_decoder=real,complete=len(records),mini_only=True,prefix_entries=0)))
                        with (folder/'stdout.json').open('x') as out,(folder/'stderr.log').open('x') as err:subprocess.run(argv,env=env,stdout=out,stderr=err,timeout=300,check=True)
                        v=json.loads((folder/'stdout.json').read_text());assert all(g['actual_output_tokens_each']==64 and g['new_input_tokens_each']==1067 for g in v['groups'])
                        records.append(dict(round=round_id,c=c,real_decoder=real,directory=str(folder)))
                        (job/'records.json').write_text(json.dumps(records,indent=2))
        (job/'result.json').write_text(json.dumps(dict(records=records,mini_only=True,full_main_unchanged=True,qualification=False,raw_independent_review_required=True),indent=2));rc=0
    except BaseException:(job/'failure.txt').write_text(traceback.format_exc())
    finally:(job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_main_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',type=Path,required=True);main(p.parse_args())
