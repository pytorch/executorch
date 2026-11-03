# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowering Hugging Face's static-cache ABI must keep both the ABI and the
model.

Tiny random decoders are exported the way Hugging Face's exporter does it (one
decode graph with a symbolic query length, the caches as inputs written in
place), specialized to a prompt chunk and a single token, rewritten, and driven
the way HF's generator drives the ABI. Greedy tokens must equal the model's
own, the inputs and outputs must be HF's, and in ExecuTorch the step must write
into the caller's cache buffers.
"""

import builtins
import importlib.util
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree

try:
    from transformers import (
        Gemma3ForCausalLM,
        Gemma3TextConfig,
        GenerationConfig,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from transformers.exporters import DynamoConfig, DynamoExporter
except ImportError:  # transformers without the exporters
    DynamoExporter = None

from executorch.backends.qualcomm.utils import hf_static
from executorch.extension.llm.io_contracts.hf_static import HfStaticDriver

L = 32  # context length
N = 4  # prompt chunk
PROMPT = [5, 17, 42, 9, 77, 3, 64]  # two chunks, the second padded
NEW_TOKENS = 8
_SMALL = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": L,
}


def _llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**_SMALL)).eval()


def _qwen3():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(head_dim=16, **_SMALL)).eval()


def _gemma3():
    torch.manual_seed(0)
    # A window at least the context: HF's ABI has no wrap-around for sliding
    # layers (their cache_position comes from a counter baked at 0).
    return Gemma3ForCausalLM(
        Gemma3TextConfig(head_dim=16, sliding_window=L, **_SMALL)
    ).eval()


@unittest.skipIf(DynamoExporter is None, "needs transformers.exporters")
class TestHfStatic(unittest.TestCase):
    make_model = staticmethod(_llama)

    def setUp(self):
        torch.set_grad_enabled(False)
        self.model = self.make_model()
        self._export_hf_graph()

    def _export_hf_graph(self):
        sample = torch.arange(4).reshape(1, 4) + 3
        self.hf_graph = (
            DynamoExporter()
            .export_for_generation(
                self.model,
                {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
                DynamoConfig(dynamic=True),
                generation_config=GenerationConfig(
                    do_sample=False, cache_implementation="static", max_cache_len=L
                ),
                keep_all_logits=True,
            )
            .components["decode"]
            .artifact
        )

    def _programs(self, **options):
        return {
            q: hf_static.to_hf_static(
                hf_static.specialize(self.hf_graph, q, L, config=self.model.config),
                **options,
            )
            for q in (1, N)
        }

    def _expected(self):
        return self.model.generate(
            torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS, do_sample=False
        )[0].tolist()

    def _generate(self, steps, layout, in_place=False):
        driver = HfStaticDriver(steps, layout, in_place=in_place)
        return driver.generate(PROMPT, NEW_TOKENS), driver

    def test_auto_attention_in_lowered_graphs(self):
        for quantized in (False, True):
            with self.subTest(quantized=quantized), patch.object(
                hf_static, "_quantize", side_effect=lambda programs, *args: programs
            ), patch.object(
                hf_static.HfStaticPartitioner, "from_options", return_value=None
            ), patch.object(
                hf_static, "to_edge_transform_and_lower", return_value=None
            ):
                _, programs, _ = hf_static._lower(
                    self.hf_graph,
                    {"decode": 1, "prefill": N},
                    L,
                    "auto",
                    quantized,
                    object() if quantized else None,
                    None,
                    ["prompt"],
                    "SM8750",
                    {},
                    {},
                    config=self.model.config,
                )
                for method, program in programs.items():
                    matmuls = [
                        n
                        for n in program.graph.nodes
                        if n.target == torch.ops.aten.matmul.default
                    ]
                    expected = 4 if method == "decode" or quantized else 2
                    self.assertEqual(
                        len(matmuls), expected * self.model.config.num_hidden_layers
                    )

    def test_abi_is_hf(self):
        programs = self._programs()
        for q, program in programs.items():
            reference = hf_static.specialize(
                self.hf_graph, q, L, config=self.model.config
            )
            self.assertEqual(
                [
                    s.arg.name
                    for s in program.graph_signature.input_specs
                    if s.kind.name == "USER_INPUT"
                ],
                [
                    s.arg.name
                    for s in reference.graph_signature.input_specs
                    if s.kind.name == "USER_INPUT"
                ],
            )
            for got, want in zip(
                pytree.tree_leaves(program.example_inputs),
                pytree.tree_leaves(reference.example_inputs),
            ):
                self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
            self.assertFalse(
                any("scaled_dot_product" in str(n.target) for n in program.graph.nodes)
            )

    def test_steps_match_hf_graph(self):
        # Same inputs through the rewritten program and HF's own graph: a padded
        # prompt chunk, then decode steps. Same logits (real rows), caches and
        # counters, element by element.
        programs = self._programs()
        references = {
            q: hf_static.specialize(self.hf_graph, q, L, config=self.model.config)
            for q in programs
        }
        layout = hf_static.HfStaticLayout.of(programs[1])
        ours = HfStaticDriver(
            {q: hf_static.flat_call(p) for q, p in programs.items()}, layout
        )
        theirs = HfStaticDriver(
            {q: hf_static.flat_call(p) for q, p in references.items()}, layout
        )
        steps = [(PROMPT[:N], 0, N), (PROMPT[N:], N, N)]
        steps += [([t], len(PROMPT) + i, 1) for i, t in enumerate([1, 2, 3])]
        for tokens, start, q in steps:
            torch.testing.assert_close(
                ours.forward(tokens, start, q), theirs.forward(tokens, start, q)
            )
            for a, b in zip(ours.state, theirs.state):
                if a.dim() == 0:
                    torch.testing.assert_close(a, b)
                else:
                    # Padding slots are not valid cache state under this contract.
                    filled = start + len(tokens)
                    torch.testing.assert_close(a[:, :, :filled], b[:, :, :filled])

    def test_greedy_matches_model(self):
        programs = self._programs()
        layout = hf_static.HfStaticLayout.of(programs[1])
        tokens, _ = self._generate(
            {q: hf_static.flat_call(p) for q, p in programs.items()}, layout
        )
        self.assertEqual(tokens, self._expected())

    def test_greedy_matches_model_masked_softmax(self):
        programs = self._programs(masked_softmax=True)
        layout = hf_static.HfStaticLayout.of(programs[1])
        tokens, _ = self._generate(
            {q: hf_static.flat_call(p) for q, p in programs.items()}, layout
        )
        self.assertEqual(tokens, self._expected())

    def test_greedy_matches_model_attention_forms(self):
        for attention, masked_softmax in itertools.product(
            ("split", "concat"), (False, True)
        ):
            programs = self._programs(
                attention=attention, masked_softmax=masked_softmax
            )
            layout = hf_static.HfStaticLayout.of(programs[1])
            tokens, _ = self._generate(
                {q: hf_static.flat_call(p) for q, p in programs.items()}, layout
            )
            self.assertEqual(tokens, self._expected())

    def test_executorch_writes_inputs_in_place(self):
        from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
        from executorch.exir.passes import MemoryPlanningPass

        programs = self._programs()
        edge = to_edge(
            {f"q{q}": p for q, p in programs.items()},
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        layout = hf_static.HfStaticLayout.of(programs[1])
        et = edge.to_executorch(
            ExecutorchBackendConfig(
                run_reinplace_pass=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        # Each cache write and counter update is an in-place kernel on the
        # method's own input value, so it lands in the buffer the caller binds.
        for plan in et.executorch_program.execution_plan:
            ops = [o.name for o in plan.operators]
            inputs = set(plan.inputs)
            writes = [
                (ops[i.instr_args.op_index], i.instr_args.args[0])
                for chain in plan.chains
                for i in chain.instructions
                if hasattr(i.instr_args, "op_index")
                and ops[i.instr_args.op_index] in ("aten::index_put_", "aten::copy_")
            ]
            windows = hf_static.sliding_windows(self.model.config)
            self.assertEqual(
                len(writes),
                3 * layout.n_layers + 2 * sum(w is not None for w in windows),
            )
            self.assertTrue(all(value in inputs for _, value in writes))

    def test_qnn_partitions(self):
        try:
            from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        except ImportError:
            self.skipTest("needs ExecuTorch")
        try:
            partitioner = hf_static.HfStaticPartitioner.from_options(
                soc_model="SM8750", use_fp16=True, shared_buffer=True
            )
        except Exception as error:  # QNN SDK not set up
            self.skipTest(f"needs the QNN SDK: {error}")
        program = self._programs()[N]
        edge = to_edge_transform_and_lower(
            program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        calls = [
            n
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function" and "getitem" not in str(n.target)
        ]
        delegates = [n for n in calls if "executorch_call_delegate" in str(n.target)]
        self.assertEqual(len(delegates), 1)
        on_cpu = {str(n.target) for n in calls if n not in delegates}
        self.assertTrue(any("index_put" in t for t in on_cpu))
        self.assertTrue(any("logical_and" in t for t in on_cpu))

    def test_export_hf_static(self):
        try:
            hf_static.HfStaticPartitioner.from_options(
                soc_model="SM8750", use_fp16=True
            )
        except Exception as error:  # QNN SDK not set up
            self.skipTest(f"needs the QNN SDK: {error}")

        class Tokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([PROMPT])}

        # Parameters require grad, as from_pretrained loads them.
        with torch.enable_grad():
            program = hf_static.export_hf_static(
                self.make_model(), Tokenizer(), "SM8750", max_seq_len=L, prefill_len=N
            )
        plans = {p.name for p in program.executorch_program.execution_plan}
        self.assertLessEqual(
            {hf_static.PREFILL_METHOD, hf_static.DECODE_METHOD, "get_io_contract"},
            plans,
        )

    def _quantizer(self, custom_annotations=()):
        try:
            from executorch.backends.qualcomm.export_utils import (
                get_backend_type,
                make_quantizer,
            )
            from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
            from torchao.quantization.pt2e import MinMaxObserver
        except Exception as error:  # QNN SDK not set up
            self.skipTest(f"needs the QNN quantizer: {error}")
        # Min/max activation observers, as the LLM recipes use.
        return make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            act_observer=MinMaxObserver,
            custom_annotations=custom_annotations,
            backend=get_backend_type("htp"),
            soc_model="SM8750",
        )

    def _quantize(self, programs, seq_mse_candidates=0, quantizer=None):
        class Tokenizer:
            def __call__(self, text):
                return {"input_ids": list(PROMPT)}

        programs = {
            hf_static.DECODE_METHOD: programs[1],
            hf_static.PREFILL_METHOD: programs[N],
        }
        lengths = {hf_static.DECODE_METHOD: 1, hf_static.PREFILL_METHOD: N}
        layout = hf_static.HfStaticLayout.of(programs[hf_static.DECODE_METHOD])
        quantized = hf_static._quantize(
            programs,
            lengths,
            layout,
            quantizer or self._quantizer(),
            Tokenizer(),
            ["hello world"],
            seq_mse_candidates,
        )
        return programs, quantized, layout

    @staticmethod
    def _steps(programs):
        return {
            1: hf_static.flat_call(programs[hf_static.DECODE_METHOD]),
            N: hf_static.flat_call(programs[hf_static.PREFILL_METHOD]),
        }

    def _quantized_fidelity(self, programs, seq_mse_candidates=0):
        """Cosine similarity of the quantized programs' logits to the float
        ones, over a prompt chunk and two decode steps."""
        programs, quantized, layout = self._quantize(programs, seq_mse_candidates)
        drivers = [
            HfStaticDriver(self._steps(p), layout) for p in (programs, quantized)
        ]
        fidelity = []
        for tokens, start, q in [(PROMPT[:N], 0, N), ([1], N, 1), ([2], N + 1, 1)]:
            ref, got = (d.forward(tokens, start, q) for d in drivers)
            fidelity.append(
                torch.nn.functional.cosine_similarity(ref, got, dim=0).item()
            )
        return fidelity

    def test_kv_8bit_annotation_keeps_float_cache_inputs(self):
        try:
            from executorch.backends.qualcomm.export_utils import (
                get_backend_type,
                make_quantizer,
            )
            from executorch.backends.qualcomm.quantizer.custom_annotation import (
                annotate_kv_8bit,
            )
            from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
        except Exception as error:  # QNN SDK not set up
            self.skipTest(f"needs the QNN quantizer: {error}")
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            custom_annotations=(annotate_kv_8bit,),
            backend=get_backend_type("htp"),
            soc_model="SM8750",
        )
        for program in self._programs(masked_softmax=True).values():
            prepared = prepare_pt2e(program.module(), quantizer)
            args, kwargs = program.example_inputs
            prepared(*args, **kwargs)
            graph = convert_pt2e(prepared).graph
            caches = [
                n
                for n in graph.nodes
                if n.op == "placeholder"
                and n.name.startswith("past_key_values_")
                and n.meta["val"].dim() == 4
            ]
            self.assertTrue(caches)
            for cache in caches:
                self.assertEqual(cache.meta["val"].dtype, torch.float32)
                self.assertIn(
                    torch.uint8,
                    {u.args[-1] for u in cache.users if "quantize" in str(u.target)},
                    cache.name,
                )

    def test_quantized_tracks_hf_graph_quantized(self):
        # The rewrite costs no accuracy under quantization: on average over the
        # steps, as close to float as HF's own graph quantized the same way.
        hf = self._quantized_fidelity(
            {
                q: hf_static.specialize(self.hf_graph, q, L, config=self.model.config)
                for q in (1, N)
            }
        )
        ours = self._quantized_fidelity(self._programs(masked_softmax=True))
        self.assertGreater(sum(ours) / len(ours), sum(hf) / len(hf) - 0.01)

    def test_quantize_with_seq_mse(self):
        fidelity = self._quantized_fidelity(self._programs(masked_softmax=True), 4)
        self.assertGreater(min(fidelity), 0.9)


@unittest.skipIf(DynamoExporter is None, "needs transformers.exporters")
class TestHfStaticSlidingWindow(unittest.TestCase):
    """Gemma3 with a sliding window shorter than the context: generation past
    the window, prompt chunks that roll it (the last one padded)."""

    WINDOW = 6
    PROMPT = [5, 17, 42, 9, 77, 3, 64, 33, 21, 8, 99]  # three chunks
    NEW_TOKENS = 20

    def setUp(self):
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        config = dict(_SMALL, num_hidden_layers=6)  # five sliding layers, one full
        self.model = Gemma3ForCausalLM(
            Gemma3TextConfig(head_dim=16, sliding_window=self.WINDOW, **config)
        ).eval()
        sample = torch.arange(4).reshape(1, 4) + 3
        self.hf_graph = (
            DynamoExporter()
            .export_for_generation(
                self.model,
                {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
                DynamoConfig(dynamic=True),
                generation_config=GenerationConfig(
                    do_sample=False, cache_implementation="static", max_cache_len=L
                ),
                keep_all_logits=True,
            )
            .components["decode"]
            .artifact
        )

    def _programs(self, **options):
        return {
            q: hf_static.to_hf_static(
                hf_static.specialize(self.hf_graph, q, L, config=self.model.config),
                **options,
            )
            for q in (1, N)
        }

    def _assert_greedy_matches_model(self, programs, steps=None, in_place=False):
        layout = hf_static.HfStaticLayout.of(programs[1])
        steps = steps or {q: hf_static.flat_call(p) for q, p in programs.items()}
        tokens = HfStaticDriver(steps, layout, in_place).generate(
            self.PROMPT, self.NEW_TOKENS
        )
        expected = self.model.generate(
            torch.tensor([self.PROMPT]),
            max_new_tokens=self.NEW_TOKENS,
            do_sample=False,
        )[0].tolist()
        # HF stops at the end-of-sequence token; past the window either way.
        self.assertGreater(len(expected), len(self.PROMPT) + 2)
        self.assertGreater(len(expected), self.WINDOW)
        self.assertEqual(tokens[: len(expected)], expected)

    def test_window_caches_keep_hf_shapes(self):
        reference = hf_static.specialize(self.hf_graph, 1, L, config=self.model.config)
        program = self._programs()[1]
        for got, want in zip(
            pytree.tree_leaves(program.example_inputs),
            pytree.tree_leaves(reference.example_inputs),
        ):
            self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
        layout = hf_static.HfStaticLayout.of(program)
        self.assertEqual(sorted({s[2] for s in layout.cache_shapes}), [self.WINDOW, L])

    def test_window_greedy_matches_model(self):
        for attention, masked_softmax in itertools.product(
            ("split", "concat"), (False, True)
        ):
            with self.subTest(attention=attention, masked_softmax=masked_softmax):
                self._assert_greedy_matches_model(
                    self._programs(attention=attention, masked_softmax=masked_softmax)
                )

    def test_window_writes_in_place_in_executorch(self):
        # The shift and the new rows are in-place kernels on the method's own
        # cache inputs; run through ExecuTorch with the caller keeping its
        # buffers, generation still matches HF.
        from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
        from executorch.exir.passes import MemoryPlanningPass

        try:
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch_from_buffer,
            )
        except ImportError:
            self.skipTest("needs ExecuTorch's runtime")
        programs = self._programs(masked_softmax=True)
        layout = hf_static.HfStaticLayout.of(programs[1])
        edge = to_edge(
            {f"q{q}": p for q, p in programs.items()},
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        et = edge.to_executorch(
            ExecutorchBackendConfig(
                run_reinplace_pass=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        module = _load_for_executorch_from_buffer(et.buffer)
        mutated = 3 * layout.n_layers

        def step(q):
            def run(*inputs):
                # The mutated inputs come back first, then the logits.
                outputs = module.run_method(f"q{q}", inputs)
                return [outputs[mutated], *outputs[:mutated]]

            return run

        self._assert_greedy_matches_model(
            programs, {q: step(q) for q in programs}, in_place=True
        )


class TestHfStaticQwen3(TestHfStatic):
    make_model = staticmethod(_qwen3)


class TestHfStaticGemma3(TestHfStatic):
    make_model = staticmethod(_gemma3)


class TestHfStaticCpuTags(unittest.TestCase):
    def test_import_without_exporters_or_pt2e(self):
        original_import = builtins.__import__

        def import_without_export(name, *args, **kwargs):
            if name in (
                "transformers.exporters",
                "torchao.quantization.pt2e.quantize_pt2e",
            ):
                raise ImportError("dependency deliberately unavailable")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            "hf_static_without_export", hf_static.__file__
        )
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=import_without_export):
            spec.loader.exec_module(module)
        self.assertIsNone(module.DynamoExporter)
        self.assertIsNone(module.prepare_pt2e)

    def test_export_prefill_window_guard_precedes_tracing(self):
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=2, sliding_window=6)
        )
        with self.assertRaisesRegex(
            ValueError, "prefill_len.*smallest sliding cache.*6"
        ):
            hf_static.export_hf_static(
                model, None, "SM8750", max_seq_len=32, prefill_len=8
            )

    def test_cpu_selection_uses_tags_not_dtype_size_or_name(self):
        graph = torch.fx.Graph()
        tokens = graph.placeholder("input_ids")
        tokens.meta["val"] = torch.zeros(1, 4, dtype=torch.int64)
        untagged = graph.call_function(
            torch.ops.aten._to_copy.default, (tokens,), {"dtype": torch.int64}
        )
        untagged.meta["val"] = torch.zeros(4, dtype=torch.int64)
        tagged = graph.call_function(torch.ops.aten.sin.default, (tokens,))
        tagged.meta = {"val": torch.zeros(128), "custom": {hf_static.CPU_META: True}}
        graph.output((untagged, tagged))
        names = hf_static._cpu_nodes(SimpleNamespace(graph=graph))
        self.assertIn(tagged.name, names)
        self.assertNotIn(untagged.name, names)


if __name__ == "__main__":
    unittest.main()
