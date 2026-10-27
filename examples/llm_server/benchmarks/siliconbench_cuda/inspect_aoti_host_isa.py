"""Inventory host instructions in existing AOTI libraries without loading them.

Run with the ExecuTorch schema/source used by the artifact. CUDA is hidden and
no library or model is executed. Existing artifact bytes remain unchanged.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--PTE',type=Path,required=True)
    p.add_argument('--expected-PTE-SHA256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    raw=a.PTE.read_bytes();assert sha(raw)==a.expected_PTE_SHA256
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(a.source.resolve().parent))
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DataLocation
    pte=deserialize_pte_binary(raw)
    assert pte.named_data and not pte.named_data.external_data
    libraries={};methods={}
    for plan in pte.program.execution_plan:
        for delegate in plan.delegates:
            if delegate.id!='CudaBackend':continue
            assert delegate.processed.location==DataLocation.INLINE
            keys=pte.program.backend_delegate_data[delegate.processed.index].data.decode().splitlines()
            assert len(keys)==2 and keys[0].endswith('_so_blob') and keys[1].endswith('_weights_blob')
            key=keys[0];entry=pte.named_data.pte_data[key]
            data=bytes(pte.named_data.buffers[entry.buffer_index])
            assert data.startswith(b'\x7fELF') and key==sha(data)+'_so_blob'
            if key in libraries:assert libraries[key]==data
            libraries[key]=data;methods[plan.name]=keys
    assert libraries
    a.output.mkdir(parents=True)
    host_info={}
    for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            if k.strip() in ('model name','flags'):host_info[k.strip()]=v.strip()
    flags=set(host_info.get('flags','').split())
    instruction=re.compile(r'^\s*([0-9a-f]+):\s*((?:[0-9a-f]{2}\s+)+)\s*([a-z][a-z0-9]*)\s*(.*)$')
    function=re.compile(r'^\s*[0-9a-f]+ <(.+)>:$')
    result=[]
    for key,data in sorted(libraries.items()):
        library=a.output/(key+'.so');library.write_bytes(data)
        r=subprocess.run(['objdump','--wide','-d','-M','intel',str(library)],capture_output=True,text=True,check=True)
        (a.output/(key+'-disassembly.log')).write_text(r.stdout)
        unsupported=[];current_function=None;count=0
        for line in r.stdout.splitlines():
            f=function.match(line)
            if f:current_function=f.group(1);continue
            i=instruction.match(line)
            if not i:continue
            count+=1;address,encoded,mnemonic,operands=i.groups()
            encoded=bytes.fromhex(encoded)
            evex=encoded[0]==0x62
            mask_instruction=bool(re.search(r'\b(?:k[0-7]|zmm[0-9]+)\b',operands)) or mnemonic.startswith(('kmov','kand','kor','kxor','knot','kshift','ktest','kortest','kunpck'))
            if (evex or mask_instruction) and 'avx512f' not in flags:
                unsupported.append({'virtual_address':int(address,16),'bytes':encoded.hex(),
                    'mnemonic':mnemonic,'operands':operands,'function':current_function,'EVEX':evex})
        assert count>0
        result.append({'so_blob_key':key,'bytes':len(data),'SHA256':sha(data),
            'disassembled_instruction_count':count,'methods':[m for m,k in methods.items() if k[0]==key],
            'unsupported_AVX512_instruction_count':len(unsupported),
            'unsupported_mnemonic_counts':dict(collections.Counter(i['mnemonic'] for i in unsupported)),
            'unsupported_instructions':unsupported})
    report={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PTE_SHA256':sha(raw),
        'PTE_source':str(a.PTE),'host_CPU':host_info,'libraries':result,'methods':methods,
        'all_libraries_preserved_without_modification':True,'no_CUDA_or_library_execution':True,
        'artifact_loading_or_correctness_pass':False,'compatibility_patch_created':False,
        'scope':'Existing x86 ELF disassembly; unsupported EVEX/mask instructions on a host without AVX512F. This is not a general CPU compatibility proof.'}
    (a.output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sha(a.PTE.read_bytes())==a.expected_PTE_SHA256
    print(json.dumps({'libraries':len(libraries),'unsupported_mnemonics':dict(collections.Counter(i['mnemonic'] for r in result for i in r['unsupported_instructions'])),
        'output':str(a.output),'model_or_GPU_not_executed':True}))


if __name__=='__main__':main()
