import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import torch
import argparse
import struct
from tqdm import tqdm
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.mediatek.neuropilot import NeuropilotPartitioner, NeuropilotQuantizer, Precision
from executorch.exir.backend.backend_details import CompileSpec
from executorch import exir
from datasets import load_dataset
from aot_utils.llm_utils.sanity_checks import *
from aot_utils.llm_utils.utils import (
    resolve_model_classes,
    get_normalized_config,
    generate_mask,
    get_dest_path,
    get_dirname,
    get_embedding_layer,
    get_exp_name,
    get_export_shapes,
    get_master_rot_emb,
    load_checkpoints,
)
from aot_utils.llm_utils.preformatter import Preformatter
import warnings
warnings.filterwarnings('ignore')

def get_argument_parser():
    parser = argparse.ArgumentParser(
        description='Run Export to ET for suppoorted LLM models.', allow_abbrev=False)
    parser.add_argument('config', type=str,
        help="[Required] Model config json file. "
        "Model config must be in same directory as all model weight bins and tokenizer files.")
    parser.add_argument('-p', '--precision', type=str, default='A16W8', choices=['A16W4', 'A16W8', 'A16W16', 'A8W4', 'A8W8'],
        help="Precision to quantize entire model to.")
    parser.add_argument('-n', '--num_chunks', type=int, default=4,
        help="Number of chunks to cut the model into. Defaults to 4.")
    parser.add_argument('-shapes', nargs='+',
        help="[Required] Expected input shapes to reconfigure TFLites to. Space separated list of "
        "shapes in the format: xtyc (e.g. 32t512c)")

    return parser

def args_sanity_checks(args):
    check_old_arg(args.config)
    check_exist(args.config, 'Config file')
    check_ext(args.config, '.json', 'Config file')
    config = get_normalized_config(args.config)
    weight_dir = get_dirname(args.config)
    check_weights_exist(weight_dir)
    check_supported_model(config)
    check_between_inclusive(args.num_chunks, 1, config.num_hidden_layers, 'num_chunks')
    check_shapes(args.shapes)

def print_args(args, exp_name):
    print("Please check if all arguments are correct:")
    print(f"Config file:                  {args.config}")
    print(f"Output tflite folder:         pte/{exp_name}")
    print(f"Quantization precision:       {args.precision}")
    print(f"Number of chunks:             {args.num_chunks}")
    print(f"Export shape(s):              {args.shapes}")
    print()

def export_to_et_ir(
    output_folder, exp_name, model, precision, max_num_token, max_cache_size, chunk_idx, export_shapes
):
    print(f"Exporting Chunk {chunk_idx} to PTE")
    example_inputs, dynamic_shapes = model.get_example_inputs(max_num_token, max_cache_size, True)

    print("Getting pre autograd ATen Dialect Graph")
    pre_autograd_aten_dialect = torch._export.capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shapes) # NOTE: Will be replaced with export
    quantizer = NeuropilotQuantizer()
    quantizer.setup_precision(getattr(Precision, precision))
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    print("Calibrating with dummy data")
    prepared_graph(*example_inputs) # dummy calibration
    converted_graph = convert_pt2e(prepared_graph, fold_quantize=False)

    print("Getting ATen Dialect Graph")
    # Fixed Shape Export Here
    for shape, ntok_and_cache in export_shapes.items():
        dest_path = get_dest_path(output_folder, exp_name, shape, chunk_idx)
        print(f"Exporting Shape: {shape} to:\n{dest_path}")
        num_token, cache_size = ntok_and_cache
        if num_token != max_num_token and cache_size != max_cache_size:
            example_inputs = model.get_example_inputs(*ntok_and_cache)
        aten_dialect: exir.ExportedProgram = torch.export.export(converted_graph, example_inputs)

        print("Lowering to Edge Dialect Graph")
        edge_program: exir.EdgeProgramManager = exir.to_edge(
            aten_dialect, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False)
        )
        del aten_dialect

        print(f"Delegating Edge Program to Neuropilot Backend")
        compile_spec = [
            CompileSpec("gno", struct.pack('3s', b"LTS")),
            CompileSpec("gno-exp", struct.pack('4s', b"True")),
            CompileSpec("gno-non-4d-tiling", struct.pack('4s', b"True")),
            CompileSpec("HighAddr", struct.pack('?', True)),
            CompileSpec("ImportForever", struct.pack('?', True))
        ]
        partitioner = NeuropilotPartitioner(compile_spec)
        delegated_program = edge_program.to_backend(partitioner)
        print(f"Exported Delegated Program:")
        print(delegated_program.exported_program())
        del edge_program

        print("Transforming delegated program to executorch backend")
        executorch_program = delegated_program.to_executorch(
            config=exir.ExecutorchBackendConfig(
                memory_planning_pass=exir.passes.MemoryPlanningPass(
                    memory_planning_algo='greedy',
                    alloc_graph_input=False,
                    alloc_graph_output=False
                ),
                extract_constant_segment=True
            )
        )

        print(f"ET Model Dest: {dest_path}\n")
        os.makedirs(dest_path.rsplit('/', 1)[0], exist_ok=True)
        with open(dest_path, "wb") as file:
            file.write(executorch_program.buffer)

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args_sanity_checks(args)
    exp_name = f"{get_exp_name(args.config)}_{args.precision}_dummy_cal_{args.num_chunks}_chunks"
    print_args(args, exp_name)

    config, weight_dir, _, chunk_class = resolve_model_classes(args.config)

    # Evenly distribute the layers across chunks.
    num_blocks_per_chunk = [
        (config.num_hidden_layers//args.num_chunks) + \
        (i < (config.num_hidden_layers % args.num_chunks)) for i in range(args.num_chunks)
    ]
    check_all_chunks_same_num_layer(num_blocks_per_chunk)

    output_folder = os.path.join('pte', exp_name)

    # Load all collected checkpoint files into one giant state_dict
    state_dict = load_checkpoints(weight_dir)

    export_shapes, max_num_token, max_cache_size = get_export_shapes(args.shapes)
    print(f"export shapes: {export_shapes}"
    )
    print(f"Max Num Token: {max_num_token}")
    print(f"Max Cache Size: {max_cache_size}")

    # Instantiate model chunks
    print("Instantiating submodels")
    models = []
    for chunk_idx, num_blocks in enumerate(num_blocks_per_chunk):
        chunk = chunk_class(
                    config,
                    num_blocks,
                    chunk_idx=chunk_idx,
                    dtype=torch.float32,
                    include_tail=(chunk_idx == args.num_chunks-1),
                    jit_trace=True
                )
        chunk = chunk.load_weights(state_dict, sum(num_blocks_per_chunk[:chunk_idx]))
        models.append(chunk)

    for chunk_idx, chunk in enumerate(models):
        export_to_et_ir(
            output_folder,
            exp_name,
            chunk,
            args.precision,
            max_num_token,
            max_cache_size,
            chunk_idx,
            export_shapes,
        )

if __name__ == "__main__":
    main()
