from torch.export import exported_program
import coremltools as ct

from llama_transformer_kvio import Transformer, ModelArgs

# Define model
import json
import torch

params_path = f"/Users/scroy/models/stories110M/params.json"
checkpoint_path = f"/Users/scroy/models/stories110M/stories110M.pt"
output_path = f"/Users/scroy/Desktop/exported2.pte"

with open(params_path, "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(
    max_seq_len=512,
    max_batch_size=1,
    use_kv_cache=False,
    use_sdpa_with_kv_cache_op=False,
    generate_full_logits=False,
    input_prune_map=None,
    output_prune_map=None,
    enable_dynamic_shape=False,
    **params,
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)

with torch.no_grad():
    model = Transformer(model_args)
    model.eval()
    model.load_state_dict(
        checkpoint,
        strict=False,
        assign=True
    )

    # [bs, n_local_kv_heads, seq_len, head_dim]
    cache_shape = (model_args.n_layers, model_args.max_batch_size, model_args.n_heads, model_args.max_seq_len, model_args.dim // model_args.n_heads)
    k_caches = torch.zeros(cache_shape, dtype=torch.float16, device="cpu")
    v_caches = torch.zeros(cache_shape, dtype=torch.float16, device="cpu")
    

    # example_inputs = (
    #     torch.ones(size=(1, model_args.max_seq_len), dtype=torch.long),
    #     torch.tensor(
    #         [0], dtype=torch.long
    #     ),  # start_pos
    #     k_caches,
    #     v_caches,
    # )

    example_inputs = (
        torch.ones(size=(1, 1), dtype=torch.long),
        torch.tensor(
            [0], dtype=torch.long
        ),  # start_pos
        k_caches,
        v_caches,
    )
    
    exported_model = torch.export.export(model, example_inputs, strict=False)


print('Exported model', exported_model)

from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program
from executorch.exir import to_edge
from executorch.exir.program._program import to_edge_with_preserved_ops

edge_config = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,
)
edge_program_manager = to_edge(exported_model, compile_config=edge_config)
print('Edge program', edge_program_manager.exported_program())

from executorch.extension.llm.export.partitioner_lib import get_coreml_partitioner
partitioner = get_coreml_partitioner(ios = 18)
delegated_edge_program_manager = edge_program_manager.to_backend(partitioner)
print('Delegated edge program', delegated_edge_program_manager.exported_program())

executorch_program = delegated_edge_program_manager.to_executorch()
with open(output_path, "wb") as file:
    executorch_program.write_to_file(file)


# # Convert to Core ML program using the Unified Conversion API.
# model_from_trace = ct.convert(
#     traced_model,
#     inputs=[ct.TensorType(shape=exi.shape) for exi in example_inputs ],
#     minimum_deployment_target=ct.target.iOS18
# )
# model_from_trace.save("/Users/scroy/Desktop/traced_model.mlpackage")


# model_from_export = ct.convert(
#     exported_model,
#     minimum_deployment_target=ct.target.iOS18
# )
# model_from_export.save("/Users/scroy/Desktop/exported_model.mlpackage")


# mlpackage = ct.convert(exported_model, minimum_deployment_target=ct.target.iOS18)

# print(mlpackage)
# mlpackage.save("/Users/scroy/Desktop/model.mlpackage")

# desc = ct.utils.MultiFunctionDescriptor()

# path = "/Users/scroy/repos/executorch/extracted_coreml_models/model_1/lowered_module"

# desc.add_function(
#     f"{path}/model_prefill.mlpackage",
#     src_function_name="main",
#     target_function_name="prefill"
# )
# desc.add_function(
#     f"{path}/model_kv.mlpackage",
#     src_function_name="main",
#     target_function_name="gen"
# )

# desc.default_function_name = "prefill"
# ct.utils.save_multifunction(desc, f"{path}/combined.mlpackage")
