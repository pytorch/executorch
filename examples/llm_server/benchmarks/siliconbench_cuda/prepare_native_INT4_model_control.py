"""Prepare a pinned public native INT4 checkpoint without changing a live server.

This only downloads data and clones a launch script with a different model
path. It does not import model code, execute inference, modify frameworks, or
accept a new checkpoint for the main comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(16*1024*1024):
            h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--expected-base-model',required=True)
    p.add_argument('--expected-group-size',type=int,required=True)
    p.add_argument('--base-wrapper',type=Path,required=True)
    p.add_argument('--base-wrapper-sha256',required=True)
    p.add_argument('--original-model-reference',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert len(a.revision)==40 and all(c in '0123456789abcdef' for c in a.revision)
    assert not a.output.exists(),'Preserve all earlier controls and model files'
    assert digest(a.base_wrapper)==a.base_wrapper_sha256
    wrapper=a.base_wrapper.read_text()
    assert wrapper.count(a.original_model_reference)==1
    a.output.mkdir(parents=True)

    def state(status,**fields):
        import os
        pending=a.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
                                          inference_not_started=True,**fields))+'\n')
        pending.replace(a.output/'progress.json')

    try:
        from huggingface_hub import HfApi,snapshot_download
        state('checking-public-checkpoint-metadata')
        info=HfApi().model_info(a.repo,revision=a.revision,files_metadata=True)
        assert info.sha==a.revision
        card=info.card_data.to_dict() if hasattr(info.card_data,'to_dict') else dict(info.card_data or {})
        base=card.get('base_model');assert a.expected_base_model==base or isinstance(base,list) and a.expected_base_model in base
        assert card.get('base_model_relation','quantized')=='quantized'
        model=a.output/'model'
        state('downloading-pinned-public-INT4-checkpoint')
        snapshot_download(repo_id=a.repo,revision=a.revision,local_dir=model,max_workers=2,
                          allow_patterns=['*.safetensors','*.safetensors.index.json','config.json','README.md',
                                          'generation_config.json','tokenizer*.json','special_tokens_map.json',
                                          'chat_template.jinja','preprocessor_config.json','processor_config.json'])
        cfg=json.loads((model/'config.json').read_text())
        assert cfg['architectures']==['MuseGlimmerForConditionalGeneration']
        q=cfg['quantization_config'];assert q['quant_method']=='compressed-tensors' and q['format']=='pack-quantized'
        groups=q['config_groups'];assert groups
        for group in groups.values():
            weights=group['weights']
            assert weights['num_bits']==4 and weights['type']=='int' and weights['group_size']==a.expected_group_size
            assert not group.get('input_activations') and not group.get('output_activations')
        state('verifying-downloaded-model-SHA256')
        files={}
        for sibling in info.siblings:
            path=model/sibling.rfilename
            if not path.is_file():continue
            sha=digest(path);lfs=sibling.lfs
            expected=getattr(lfs,'sha256',None) if lfs is not None else None
            if isinstance(lfs,dict):expected=lfs.get('sha256')
            if path.suffix=='.safetensors':
                assert expected and sha==expected,'Public weight checksum mismatch'
            files[sibling.rfilename]=dict(bytes=path.stat().st_size,sha256=sha,public_LFS_sha256=expected)
        assert any(name.endswith('.safetensors') for name in files)
        copied=wrapper.replace(a.original_model_reference,shlex.quote(str(model.resolve())))
        launch=a.output/'serve-native-INT4-control.sh';launch.write_text(copied);launch.chmod(0o700)
        import subprocess
        subprocess.run(['bash','-n',str(launch)],check=True,timeout=30)
        assert digest(a.base_wrapper)==a.base_wrapper_sha256
        provenance=dict(repo=a.repo,revision=a.revision,base_model=a.expected_base_model,
                        model=str(model.resolve()),model_files=files,quantization=q,
                        base_wrapper=str(a.base_wrapper),base_wrapper_sha256=a.base_wrapper_sha256,
                        wrapper=str(launch.resolve()),wrapper_sha256=digest(launch),
                        original_wrapper_unchanged=True,only_model_path_changed=True,
                        framework_code_unchanged=True,tokenizer_and_request_flags_unchanged=True,
                        new_checkpoint_not_accepted_for_main=True,inference_not_started=True)
        (a.output/'model-control-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        state('completed',model_control_prepared=True)
    except Exception as exc:
        state('failed',error_type=type(exc).__name__,detail=str(exc)[-1000:])
        raise


if __name__=='__main__':
    main()
