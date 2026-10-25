"""Prepare an isolated HTTP source overlay for an exactly received native worker.

Retain the original Python source and native build. Match the already qualified
HTTP stop handling with two explicit source transformations, then verify the
complete transport source hashes. This preparation does no model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branch(src,dst,expanded):
    dst.mkdir(parents=True,exist_ok=False)
    for child in src.iterdir():
        if child.name in expanded:continue
        (dst/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','worker','output','expected-python-hashes']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--local-qualified-provenance',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    src=a.source.resolve();dst=a.output/'executorch'
    branch(src,dst,{'examples'})
    branch(src/'examples',dst/'examples',{'llm_server'})
    branch(src/'examples/llm_server',dst/'examples/llm_server',{'python'})
    shutil.copytree(src/'examples/llm_server/python',dst/'examples/llm_server/python',
                    ignore=shutil.ignore_patterns('__pycache__'))
    transformations={
        'serving_chat.py':('gen_stops = self._stops + self._content_specials + self._request_stops(req)',
                           'gen_stops = self._stops + self._request_stops(req)'),
        'chat_template.py':('    "<turn|>",\n    "</s>",','    "<turn|>",\n    "<|tool_response>",\n    "</s>",')}
    expected=json.loads(a.expected_python_hashes.read_text())
    for name,(old,new) in transformations.items():
        f=dst/'examples/llm_server/python'/name
        if sha(f)==expected[name]:continue
        text=f.read_text()
        if text.count(old)==1:changed=text.replace(old,new)
        else:assert text.count(new)==1,name;changed=text.replace(new,old)
        f.write_text(changed)
    actual={}
    for name,wanted in expected.items():
        f=(dst/'examples/models/gemma4_31b/serve.py' if name=='Gemma.serve.py'
           else dst/'examples/llm_server/python'/name)
        actual[name]=sha(f);assert actual[name]==wanted,(name,actual[name],wanted)
    provenance_path=a.local_qualified_provenance or a.worker/'build-provenance.json'
    if a.local_qualified_provenance:
        origin=json.loads(provenance_path.read_text())
        assert origin['qualified_core_SHA256']=='fae8d0f4d55051437b646a379cb3c6751ccd549537dcc4040281a98087eb232e'
    else:
        transfer=json.loads((a.worker/'transfer-provenance.json').read_text())
        assert transfer['every_received_file_SHA256_verified']
        origin=json.loads(provenance_path.read_text())
    binary=a.worker/'gemma_runtime_columns_worker'
    assert sha(binary)==origin['binary_SHA256']
    env=__import__('os').environ.copy()
    env['LD_LIBRARY_PATH']=str(a.worker)+':'+env.get('LD_LIBRARY_PATH','')
    result=subprocess.run(['ldd',str(binary)],env=env,capture_output=True,text=True,check=True)
    assert 'not found' not in result.stdout,result.stdout
    original_hash=sha(provenance_path)
    provenance={**origin,'binary':str(binary),'received_binary_exactly_qualified':True,
                'origin_provenance':origin,'origin_provenance_SHA256':original_hash,
                'matched_HTTP_Python_hashes':actual,'ldd':result.stdout}
    (a.output/'build-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (a.output/'progress.json').write_text(json.dumps(dict(status='CPU-HTTP-worker-ready-pending-model',
         original_source_unmodified=True,Python_source=str(dst),GPU_validation_pending=True),indent=2)+'\n')
    print('TASK_JSON:'+json.dumps(dict(prepared=True,Python_source=str(dst),native_binary_SHA256=sha(binary))))


if __name__=='__main__':main()
