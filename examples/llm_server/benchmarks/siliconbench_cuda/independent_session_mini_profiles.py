"""Explicit model geometry for mini-only lifecycle and timing qualification."""

PROFILES = {
    'muse-glimmer-first4': dict(layers=4, vocab=202048, max_step=2048,
        window=2048, ring_layers=3,
        GGUF_SHA256='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'),
    'gemma4-first6': dict(layers=6, vocab=262144, max_step=512,
        window=1024, ring_layers=5,
        GGUF_SHA256='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'),
}
DEFAULT_PROFILE = 'muse-glimmer-first4'

FULL_PROFILES = {
    'muse-glimmer-52layers': dict(PROFILES['muse-glimmer-first4'], layers=52,
                                 ring_layers=39),
    'gemma4-48layers': dict(PROFILES['gemma4-first6'], layers=48,
                           ring_layers=40),
}


def validate_full_audit(audit, profile, independent):
    expected = FULL_PROFILES[profile]
    assert audit['mini_is_not_full_model'] is False
    assert audit['retained_original_layer_indices'] == list(range(expected['layers']))
    assert audit['independent_capacity'] == independent
    assert audit['GGUF_sha256'] == expected['GGUF_SHA256']
    assert audit['initial_runtime_KV'] == 512 and audit['logical_context'] == 131072
    assert audit['declared_owner_limit'] == 21 and audit['selected_logit_limit'] == 64
    assert audit['static_forwards'] == [1, 2, 4, 8, 16, 32, 64]
    metadata = audit['metadata']
    assert metadata['get_n_caches'] == expected['layers']
    assert metadata['get_vocab_size'] == expected['vocab']
    assert metadata['get_max_context_len'] == 131072
    assert metadata['get_max_prefill_chunk'] == expected['max_step']
    assert metadata['get_activation_dtype'] == (15 if profile=='gemma4-48layers' else 'bfloat16')
    assert metadata['get_offgraph_kv_sequence_info_abi'] == (3 if independent else 2)
    manifests = audit['offgraph_manifests']
    assert set(manifests) == {f'forward_{m}' for m in (1,2,4,8,16,32,64)} | {'forward_others'}
    for manifest in manifests.values():
        assert manifest['runtime_columns'] and manifest['maximum_capacity'] == 131072
        assert manifest.get('independent_capacity', False) == independent
        assert manifest['max_write'] == expected['max_step'] and manifest['max_seqs'] == 21
        layers = manifest['layers']
        assert len(layers) == expected['layers']
        assert sum(layer['policy']=='ring' for layer in layers) == expected['ring_layers']
        assert sum(layer['policy']=='flat' for layer in layers) == expected['layers']-expected['ring_layers']
        assert all(layer['window']==expected['window'] for layer in layers if layer['policy']=='ring')
    return expected


def validate_mini_audit(audit, profile, independent):
    expected = PROFILES[profile]
    assert audit['mini_is_not_full_model']
    assert audit['retained_original_layer_indices'] == list(range(expected['layers']))
    assert audit['independent_capacity'] == independent
    assert audit['GGUF_sha256'] == expected['GGUF_SHA256']
    assert audit['initial_runtime_KV'] == 512 and audit['logical_context'] == 131072
    assert audit['declared_owner_limit'] == 21 and audit['selected_logit_limit'] == 64
    assert audit['static_forwards'] == [1, 2, 4, 8, 16, 32, 64]
    metadata = audit['metadata']
    assert metadata['get_n_caches'] == expected['layers']
    assert metadata['get_vocab_size'] == expected['vocab']
    assert metadata['get_max_prefill_chunk'] == expected['max_step']
    # Gemma emits the ExecuTorch ScalarType enum; MG emits its dtype name.
    # schema/scalar_type.fbs and portable_type/scalar_type.h define BF16=15.
    activation = metadata['get_activation_dtype']
    if profile == 'gemma4-first6':
        assert type(activation) is int and activation == 15
    else:
        assert activation == 'bfloat16'
    assert metadata['get_max_context_len'] == 131072
    assert metadata['get_offgraph_kv_sequence_info_abi'] == (3 if independent else 2)
    manifests = audit['offgraph_manifests']
    assert set(manifests) == {f'forward_{m}' for m in (1,2,4,8,16,32,64)} | {'forward_others'}
    for manifest in manifests.values():
        assert manifest['runtime_columns'] and manifest['maximum_capacity'] == 131072
        assert manifest.get('independent_capacity', False) == independent
        assert manifest['max_write'] == expected['max_step'] and manifest['max_seqs'] == 21
        layers = manifest['layers']
        assert len(layers) == expected['layers']
        assert sum(layer['policy'] == 'ring' for layer in layers) == expected['ring_layers']
        assert sum(layer['policy'] == 'flat' for layer in layers) == 1
        assert all(layer['window'] == expected['window'] for layer in layers if layer['policy'] == 'ring')
    return expected
