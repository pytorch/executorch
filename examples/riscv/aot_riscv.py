# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AOT export for the RISC-V smoke tests.

Exports the model selected by ``--model`` to a BundledProgram (.bpte) that
either ``executor_runner`` (linux) or ``executor_runner_baremetal`` (qemu
virt + semihosting) consumes. The bundled-IO comparison path inside the
runner emits ``Test_result: PASS`` per testset, which is what run.sh greps.
"""

import argparse
import logging
from pathlib import Path

import torch
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def build_add():
    model = AddModule().eval()
    example_inputs = (torch.ones(1, 4), torch.full((1, 4), 2.0))
    test_inputs = [
        (torch.ones(1, 4), torch.full((1, 4), 2.0)),
        (torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)),
    ]
    return model, example_inputs, test_inputs, True


def build_mv2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    torch.manual_seed(0)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    test_inputs = [example_inputs]
    return model, example_inputs, test_inputs, False


def build_mobilebert():
    from transformers import MobileBertConfig, MobileBertModel

    config = MobileBertConfig(
        vocab_size=1024,
        hidden_size=128,
        embedding_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        intra_bottleneck_size=32,
    )

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MobileBertModel(config).eval()

        def forward(self, input_ids):
            return self.model(input_ids).last_hidden_state

    model = Wrapper().eval()
    example_inputs = (torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),)
    test_inputs = [example_inputs]
    return model, example_inputs, test_inputs, False


def build_llama2():
    # Use the executorch native Transformer (matches MODEL_NAME_TO_MODEL["llama2"]
    # in examples/models/__init__.py). Unlike HF LlamaModel, RoPE freqs are
    # precomputed buffers and just sliced at forward time, so no
    # torch.arange()/Long causal mask is built per forward — which is what
    # the PT2E XNNPACK quantizer trips over on HF Llama.
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    seq_len = 8
    args = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # GQA: kv_heads < n_heads exercises the GQA path
        vocab_size=1024,
        hidden_dim=256,  # SwiGLU FFN: gate + up projections at this width
        max_seq_len=seq_len,
        max_context_len=seq_len,
        rope_theta=10000.0,
    )
    torch.manual_seed(0)
    model = construct_transformer(args).eval()
    example_inputs = (torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),)
    test_inputs = [example_inputs]
    return model, example_inputs, test_inputs, False


def build_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    torch.manual_seed(0)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    test_inputs = [example_inputs]
    return model, example_inputs, test_inputs, False


def build_yolo26():
    # Mirrors examples/models/yolo26/export_and_validate.py: predict() once
    # to materialise the predictor state Ultralytics expects pre-export.
    import numpy as np
    from ultralytics import YOLO

    input_h, input_w = 320, 320
    yolo = YOLO("yolo26n")
    yolo.predict(
        np.ones((input_h, input_w, 3)),
        imgsz=(input_h, input_w),
        device="cpu",
    )

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = yolo.model.to(torch.device("cpu")).eval()

        def forward(self, x):
            # yolo.model emits (predictions, feature_maps) in eval; keep the
            # predictions tensor so BundledIO sees a single tensor output.
            out = self.model(x)
            return out[0] if isinstance(out, (tuple, list)) else out

    model = Wrapper().eval()
    torch.manual_seed(0)
    example_inputs = (torch.randn(1, 3, input_h, input_w),)
    test_inputs = [example_inputs]
    return model, example_inputs, test_inputs, False


MODELS = {
    "add": build_add,
    "mv2": build_mv2,
    "mobilebert": build_mobilebert,
    "llama2": build_llama2,
    "resnet18": build_resnet18,
    "yolo26": build_yolo26,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="add",
        help="Which model to export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .bpte path (default: <model>_riscv.bpte)",
    )
    parser.add_argument(
        "--backend",
        choices=("portable", "xnnpack"),
        default="portable",
        help="AOT backend: 'portable' runs everything on the portable kernels, "
        "'xnnpack' adds the XNNPACK partitioner (default: portable)",
    )
    parser.add_argument(
        "--os",
        choices=("linux", "baremetal"),
        default="linux",
        help="Target OS for the runner that will consume this .bpte. The .bpte "
        "itself is OS-independent; the flag is logged so callers can verify "
        "the AOT/runtime sides agree (default: linux)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Produce an 8-bit quantized model",
    )
    parser.add_argument(
        "--debug-xnnpack",
        action="store_true",
        help="Enable XNNPACK partitioner DEBUG logging and dump the lowered graph",
    )
    args = parser.parse_args()

    if args.debug_xnnpack and args.backend != "xnnpack":
        parser.error("--debug-xnnpack requires --backend=xnnpack")

    # xnnpack pulls in pthreads + dynamic loading; baremetal runner doesn't have those.
    if args.os == "baremetal" and args.backend == "xnnpack":
        parser.error("--backend=xnnpack is not supported on --os=baremetal")

    if args.debug_xnnpack:
        logging.basicConfig(level=logging.DEBUG)

    if args.output is None:
        args.output = Path(f"{args.model}_riscv.bpte")

    model, example_inputs, test_inputs, strict = MODELS[args.model]()

    if args.quantize:
        from executorch.examples.xnnpack import MODEL_NAME_TO_OPTIONS, QuantType
        from executorch.examples.xnnpack.quantization.utils import quantize

        if args.model not in MODEL_NAME_TO_OPTIONS:
            parser.error(f"No XNNPACK quantization recipe for model {args.model!r}")
        quant_type = MODEL_NAME_TO_OPTIONS[args.model].quantization
        if quant_type == QuantType.NONE:
            parser.error(f"Quantization recipe for {args.model!r} is NONE")
        ep = export(model, example_inputs, strict=strict)
        model = quantize(ep.module(), example_inputs, quant_type)

    exported = export(model, example_inputs, strict=strict)
    partitioners = []
    if args.backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        partitioners.append(XnnpackPartitioner(verbose=args.debug_xnnpack))

    compile_config = None
    if args.quantize:
        from executorch.exir import EdgeCompileConfig

        compile_config = EdgeCompileConfig(_check_ir_validity=False)

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=partitioners,
        compile_config=compile_config,
    )
    delegated = sum(
        1
        for n in edge.exported_program().graph.nodes
        if n.op == "call_function" and "call_delegate" in str(n.target)
    )
    print(
        f"[aot_riscv] model={args.model} backend={args.backend} os={args.os} "
        f"quantize={args.quantize} delegated_nodes={delegated}"
    )

    if args.debug_xnnpack:
        from executorch.exir.backend.utils import print_delegated_graph

        print_delegated_graph(edge.exported_program().graph_module)

    et_program = edge.to_executorch()

    test_suite = MethodTestSuite(
        method_name="forward",
        test_cases=[
            MethodTestCase(inputs=inp, expected_outputs=(model(*inp),))
            for inp in test_inputs
        ],
    )

    bundled = BundledProgram(et_program, [test_suite])
    serialized = serialize_from_bundled_program_to_flatbuffer(bundled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    print(f"Wrote {args.output} ({len(serialized)} bytes)")


if __name__ == "__main__":
    main()
