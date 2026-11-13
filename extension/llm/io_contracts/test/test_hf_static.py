# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the registered hf_static runner on real tiny HF and ET programs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.llm.export import hf_static
from executorch.extension.llm.io_contracts.hf_static import (
    constant_method,
    HfStaticDriver,
    HfStaticRunner,
    tensor_dtype,
)
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

try:
    from transformers import GenerationConfig, MistralConfig, MistralForCausalLM
    from transformers.exporters import (
        DynamoConfig,
        DynamoExporter,
        ExecutorchModelRunner,
        ExportedGenerator,
    )
    from transformers.exporters.io_contracts import io_contract_runner
except ImportError:
    DynamoExporter = None


class TestRuntimeMetadata(unittest.TestCase):
    def test_tensor_dtype_accepts_runtime_codes_and_torch_dtypes(self):
        for code, dtype in (
            (0, torch.uint8),
            (1, torch.int8),
            (2, torch.int16),
            (3, torch.int32),
            (4, torch.int64),
            (5, torch.float16),
            (6, torch.float32),
            (7, torch.float64),
            (11, torch.bool),
            (15, torch.bfloat16),
            (27, torch.uint16),
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    tensor_dtype(SimpleNamespace(dtype=lambda code=code: code)), dtype
                )
                self.assertEqual(
                    tensor_dtype(SimpleNamespace(dtype=lambda dtype=dtype: dtype)),
                    dtype,
                )
        with self.assertRaisesRegex(ValueError, "Unsupported tensor dtype code"):
            tensor_dtype(SimpleNamespace(dtype=lambda: 99))

    def test_constant_method_preserves_default_and_output_cardinality(self):
        program = SimpleNamespace(method_names={"metadata"})
        sentinel = object()
        self.assertIs(constant_method(program, "absent", sentinel), sentinel)
        for values, expected in (([], []), ([1], 1), ([1, 2], [1, 2])):
            with self.subTest(values=values):
                program.load_method = lambda name, values=values: SimpleNamespace(
                    execute=lambda inputs, values=values: values
                )
                self.assertEqual(constant_method(program, "metadata"), expected)


class _PortableProgram:
    def __init__(self, buffer):
        self.module = _load_for_executorch_from_buffer(buffer)
        self.method_names = set(self.module.method_names())

    def load_method(self, name):
        return SimpleNamespace(
            metadata=self.module.method_meta(name),
            execute=lambda inputs: self.module.run_method(name, inputs),
        )


class _Program:
    def __init__(self, program, constants=None, methods=None):
        self.program = program
        self.constants = constants or {}
        self.methods = methods or {}
        self.method_names = program.method_names

    def load_method(self, name):
        if name in self.constants:
            return SimpleNamespace(execute=lambda inputs: [self.constants[name]])
        return self.methods.get(name) or self.program.load_method(name)


@unittest.skipIf(DynamoExporter is None, "needs transformers.exporters")
class TestHfStaticRunner(unittest.TestCase):
    CONTEXT = 32
    WINDOW = 6
    CHUNK = 4
    PROMPT = [5, 17, 42, 9, 77, 3, 64, 33, 21, 8, 99]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = MistralForCausalLM(
            MistralConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=128,
                max_position_embeddings=cls.CONTEXT,
                sliding_window=cls.WINDOW,
                eos_token_id=None,
            )
        ).eval()
        sample = torch.tensor([[3, 4, 5, 6]])
        cls.generation_config = GenerationConfig(
            do_sample=False,
            cache_implementation="static",
            max_cache_len=cls.CONTEXT,
        )
        with torch.no_grad():
            artifacts = DynamoExporter().export_for_generation(
                cls.model,
                {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
                DynamoConfig(dynamic=True),
                generation_config=cls.generation_config,
                keep_all_logits=True,
            )
            cls.hf_graph = artifacts.components["decode"].artifact
            cls.metadata = artifacts.components["decode"].metadata.raw
            cls.programs = {
                name: hf_static.to_hf_static(
                    hf_static.specialize(
                        cls.hf_graph, q, cls.CONTEXT, config=cls.model.config
                    )
                )
                for name, q in (("decode", 1), ("prefill", cls.CHUNK))
            }
        constants = {
            "get_io_contract": "hf_static",
            "get_io_contract_version": 1,
            "get_max_seq_len": cls.CONTEXT,
            "get_n_layers": 2,
            "get_cache_dtype": "float32",
            "get_eos_ids": [],
        }
        cls.et = to_edge(
            cls.programs,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            constant_methods=constants,
        ).to_executorch(
            ExecutorchBackendConfig(
                run_reinplace_pass=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        cls.buffer = cls.et.buffer

    def _load(self):
        return _PortableProgram(self.buffer)

    def _expected(self, prompt, count):
        with torch.no_grad():
            return self.model.generate(
                torch.tensor([prompt]), max_new_tokens=count, do_sample=False
            )[0, len(prompt) :].tolist()

    def test_all_sliding_mistral_shapes_and_greedy(self):
        layout = hf_static.HfStaticLayout.of(self.programs["decode"])
        self.assertEqual(layout.max_seq_len, self.CONTEXT)
        self.assertEqual([s[2] for s in layout.cache_shapes], [self.WINDOW] * 4)
        runner = HfStaticRunner(self._load())
        self.assertEqual(
            runner.generate(self.PROMPT, 20), self._expected(self.PROMPT, 20)
        )

    def test_native_hf_loader_and_exported_generator(self):
        runner = ExecutorchModelRunner._load(self.buffer, self.metadata)
        generator = ExportedGenerator(
            {"decoder": runner}, self.model.config, self.generation_config
        )
        for prompt in (self.PROMPT, [11]):
            with self.subTest(prompt=prompt):
                output = generator.generate(
                    input_ids=torch.tensor([prompt]), max_new_tokens=8, do_sample=False
                )
                self.assertEqual(
                    output[0, len(prompt) :].tolist(), self._expected(prompt, 8)
                )

    def test_exported_generator_uses_registered_runner(self):
        runner = io_contract_runner(
            "hf_static", "executorch", self._load(), export_metadata=self.metadata
        )
        self.assertTrue(runner.drives_generation)
        generator = ExportedGenerator(
            {"decoder": runner}, self.model.config, self.generation_config
        )
        for prompt in (self.PROMPT, [11]):
            with self.subTest(prompt=prompt):
                result = generator.generate(
                    input_ids=torch.tensor([prompt]), max_new_tokens=8, do_sample=False
                )
                self.assertEqual(
                    result[0, len(prompt) :].tolist(), self._expected(prompt, 8)
                )

    def test_exported_generator_rejects_caller_padding_and_custom_positions(self):
        runner = io_contract_runner(
            "hf_static", "executorch", self._load(), export_metadata=self.metadata
        )
        generator = ExportedGenerator(
            {"decoder": runner}, self.model.config, self.generation_config
        )
        ids = torch.tensor([[0, *self.PROMPT]])
        mask = torch.tensor([[0, *([1] * len(self.PROMPT))]])
        with self.assertRaisesRegex(ValueError, "caller-padded"):
            generator.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=2, do_sample=False
            )
        runner.reset_state()
        with self.assertRaisesRegex(ValueError, "consecutive absolute"):
            runner(
                input_ids=torch.tensor([self.PROMPT]),
                position_ids=torch.arange(len(self.PROMPT))[None] + 1,
            )

    def test_padding_rows_are_false_when_window_equals_context(self):
        captured = []
        layout = hf_static.HfStaticLayout(
            [
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values_0",
                "past_key_values_1",
                "past_key_values_2",
            ],
            4,
            32,
            1,
            [torch.Size([1, 1, 32, 2])] * 2,
            [torch.Size([1, 1, 32, 2]), torch.Size([1, 1, 32, 2]), torch.Size([])],
            torch.float32,
            {"attention_mask": 32},
            state_dtypes=[torch.float32, torch.float32, torch.int64],
        )

        def step(*inputs):
            captured.append(inputs[1])
            return [torch.zeros(1, 4, 2), *inputs[-3:]]

        driver = HfStaticDriver({4: step}, layout)
        driver.forward([1, 2, 3], 0, 4)
        self.assertFalse(bool(captured[0][:, :, 3].any()))
        self.assertTrue(bool(captured[0][:, :, :3].any()))

    def test_registered_runner_keeps_cache_buffers_across_steps(self):
        runner = HfStaticRunner(self._load())
        pointers = [tensor.data_ptr() for tensor in runner.driver.state]
        runner.forward(self.PROMPT)
        self.assertEqual(
            pointers, [tensor.data_ptr() for tensor in runner.driver.state]
        )
        runner.forward([121])
        self.assertEqual(
            pointers, [tensor.data_ptr() for tensor in runner.driver.state]
        )

    def test_reset_and_padded_chunk_counters(self):
        runner = HfStaticRunner(self._load())
        first = runner.generate(self.PROMPT, 8)
        self.assertEqual(
            [t.item() for t in runner.driver.state if t.dim() == 0],
            [len(self.PROMPT) + 8] * 2,
        )
        pointers = [t.data_ptr() for t in runner.driver.state]
        runner.reset()
        self.assertEqual(
            [t.item() for t in runner.driver.state if t.dim() == 0], [0, 0]
        )
        self.assertEqual(pointers, [t.data_ptr() for t in runner.driver.state])
        self.assertEqual(runner.generate(self.PROMPT, 8), first)

    def test_prepare_overwrites_stale_counters(self):
        runner = HfStaticRunner(self._load())
        for tensor in runner.driver.state:
            if tensor.dim() == 0:
                tensor.fill_(999)
        runner.forward(self.PROMPT)
        self.assertEqual(
            [t.item() for t in runner.driver.state if t.dim() == 0],
            [len(self.PROMPT)] * 2,
        )

    def test_near_context_tail_uses_decode(self):
        runner = HfStaticRunner(self._load())
        prefix = (self.PROMPT * 3)[:29]
        runner.forward(prefix)
        calls = []
        for q, step in list(runner.driver.steps.items()):

            def record(*inputs, q=q, step=step):
                calls.append(q)
                return step(*inputs)

            runner.driver.steps[q] = record
        logits = runner.forward([3, 4])
        self.assertEqual(calls, [1, 1])
        self.assertEqual(runner.filled, 31)
        expected = self._expected(prefix + [3, 4], 1)[0]
        self.assertEqual(int(logits.argmax()), expected)
        runner.forward([expected])
        with self.assertRaisesRegex(ValueError, "remaining context"):
            runner.forward([1])

    def test_cache_aliases_and_no_functional_cache_copy(self):
        for plan in self.et.executorch_program.execution_plan:
            if plan.name not in ("prefill", "decode"):
                continue
            self.assertEqual(sorted(plan.outputs[:6]), sorted(plan.inputs[-6:]))
            ops = [o.name for o in plan.operators]
            self.assertNotIn("aten::index_put", ops)
            self.assertNotIn("aten::index_copy", ops)
            instructions = [
                i.instr_args
                for c in plan.chains
                for i in c.instructions
                if hasattr(i.instr_args, "op_index")
            ]
            writes = [
                i
                for i in instructions
                if ops[i.op_index] in ("aten::index_put_", "aten::index_copy_")
            ]
            self.assertEqual(len(writes), 8)
            for write in writes:
                self.assertIn(write.args[0], plan.inputs)
            copies = [i for i in instructions if ops[i.op_index] == "aten::copy_"]
            self.assertEqual(len(copies), 2)
            for copy in copies:
                self.assertEqual(plan.values[copy.args[0]].val.sizes, [])
        layout = hf_static.HfStaticLayout.of(self.programs["decode"])
        module = _load_for_executorch_from_buffer(self.buffer)
        steps = {}
        for q, method_name in ((1, "decode"), (4, "prefill")):

            def step(*inputs, method_name=method_name):
                outputs = module.run_method(method_name, inputs, clone_outputs=False)
                self.assertEqual(len(outputs), 13)
                self.assertEqual(
                    sorted(t.data_ptr() for t in inputs[-6:]),
                    sorted(t.data_ptr() for t in outputs[:6]),
                )
                return [outputs[6], *outputs[7:]]

            steps[q] = step
        driver = HfStaticDriver(steps, layout, in_place=True)
        self.assertEqual(
            driver.generate(self.PROMPT, 8)[len(self.PROMPT) :],
            self._expected(self.PROMPT, 8),
        )

    def test_rejects_newer_or_invalid_versions(self):
        for version in (2, 0, None, True):
            with self.subTest(version=version), self.assertRaisesRegex(
                ValueError, "contract version"
            ):
                HfStaticRunner(
                    _Program(self._load(), {"get_io_contract_version": version})
                )

    def test_rejects_wrong_cache_dtype(self):
        for dtype in ("float16", "uint16", None):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                ValueError, "get_cache_dtype"
            ):
                HfStaticRunner(_Program(self._load(), {"get_cache_dtype": dtype}))

    def test_rejects_mismatched_layer_or_context_metadata(self):
        for constants in (
            {"get_n_layers": 3},
            {"get_max_seq_len": 4},
            {"get_n_layers": 0},
        ):
            with self.subTest(constants=constants), self.assertRaises(ValueError):
                HfStaticRunner(_Program(self._load(), constants))

    def test_cpu_tags_survive_export_and_to_edge(self):
        edge = to_edge(
            self.programs, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        for name in self.programs:
            hf_static.tag_cache_mutations(edge.exported_program(name))
            graph = edge.exported_program(name).graph
            writes = [n for n in graph.nodes if "index_put" in str(n.target)]
            self.assertEqual(len(writes), 8)
            self.assertTrue(
                all(n.meta.get("custom", {}).get(hf_static.CPU_META) for n in writes)
            )
            booleans = [
                n
                for n in graph.nodes
                if n.op == "call_function"
                and isinstance(n.meta.get("val"), torch.Tensor)
                and n.meta["val"].dtype == torch.bool
            ]
            self.assertTrue(booleans)
            self.assertTrue(
                all(n.meta.get("custom", {}).get(hf_static.CPU_META) for n in booleans)
            )

    def test_partial_attention_match_is_rejected(self):
        program = hf_static.specialize(
            self.hf_graph, 1, self.CONTEXT, config=self.model.config
        )
        original = hf_static._cache_attention
        with patch.object(
            hf_static, "_cache_attention", side_effect=lambda graph: original(graph)[:1]
        ):
            with self.assertRaisesRegex(ValueError, "Expected 2.*matched 1"):
                hf_static.to_hf_static(program)

    def test_rejects_malformed_prefill_inputs(self):
        for index, shape, dtype in (
            (0, (1, 4), torch.int32),
            (1, (1, 1, 1, 6), torch.bool),
            (2, (1, 1), torch.int64),
            (2, (1, 4), torch.int32),
            (3, (1, 1, 6, 32), torch.float32),
            (5, (), torch.int32),
        ):
            program = self._load()
            method = program.load_method("prefill")
            original = method.metadata

            class Metadata:
                def __init__(self, original, index, shape, dtype):
                    self.original, self.index = original, index
                    self.shape, self.dtype = shape, dtype

                def __getattr__(self, name):
                    return getattr(self.original, name)

                def input_tensor_meta(self, i):
                    return (
                        SimpleNamespace(
                            sizes=lambda: self.shape, dtype=lambda: self.dtype
                        )
                        if i == self.index
                        else self.original.input_tensor_meta(i)
                    )

            bad_method = SimpleNamespace(
                metadata=Metadata(original, index, shape, dtype)
            )
            with self.subTest(index=index), self.assertRaises(ValueError):
                HfStaticRunner(_Program(program, methods={"prefill": bad_method}))

    def test_rejects_int32_tokens_or_positions_when_methods_agree(self):
        program = self._load()
        for index in (0, 2):

            class Metadata:
                def __init__(self, original, index):
                    self.original, self.index = original, index

                def __getattr__(self, name):
                    return getattr(self.original, name)

                def input_tensor_meta(self, i):
                    info = self.original.input_tensor_meta(i)
                    return (
                        SimpleNamespace(sizes=info.sizes, dtype=lambda: torch.int32)
                        if i == self.index
                        else info
                    )

            methods = {
                name: SimpleNamespace(
                    metadata=Metadata(program.load_method(name).metadata, index)
                )
                for name in ("decode", "prefill")
            }
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "input_ids|position_ids"
            ):
                HfStaticRunner(_Program(program, methods=methods))

    def test_rejects_float64_even_with_matching_cache_metadata(self):
        program = self._load()

        class Metadata:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def input_tensor_meta(self, i):
                info = self.original.input_tensor_meta(i)
                return (
                    SimpleNamespace(sizes=info.sizes, dtype=lambda: torch.float64)
                    if i in (3, 4, 6, 7)
                    else info
                )

            def output_tensor_meta(self, i):
                info = self.original.output_tensor_meta(i)
                return (
                    SimpleNamespace(sizes=info.sizes, dtype=lambda: torch.float64)
                    if len(info.sizes()) == 4
                    else info
                )

        methods = {
            name: SimpleNamespace(metadata=Metadata(program.load_method(name).metadata))
            for name in ("decode", "prefill")
        }
        with self.assertRaisesRegex(ValueError, "float cache inputs"):
            HfStaticRunner(_Program(program, {"get_cache_dtype": "float64"}, methods))

    def test_rejects_memory_planned_cache_state(self):
        planned = to_edge(
            self.programs,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            constant_methods={
                "get_io_contract": "hf_static",
                "get_io_contract_version": 1,
                "get_max_seq_len": self.CONTEXT,
                "get_n_layers": 2,
                "get_cache_dtype": "float32",
            },
        ).to_executorch(
            ExecutorchBackendConfig(
                run_reinplace_pass=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            )
        )
        with self.assertRaisesRegex(ValueError, "unplanned CPU buffers"):
            HfStaticRunner(_PortableProgram(planned.buffer))

    def test_rejects_device_cache_state(self):
        program = self._load()
        method = program.load_method("decode")
        original = method.metadata

        class Metadata:
            def __getattr__(self, name):
                return getattr(original, name)

            def input_tensor_meta(self, i):
                info = original.input_tensor_meta(i)
                if i == original.num_inputs() - 6:
                    return SimpleNamespace(
                        sizes=info.sizes,
                        dtype=info.dtype,
                        is_memory_planned=info.is_memory_planned,
                        device=lambda: "cuda:0",
                    )
                return info

        with self.assertRaisesRegex(ValueError, "unplanned CPU buffers"):
            HfStaticRunner(
                _Program(
                    program, methods={"decode": SimpleNamespace(metadata=Metadata())}
                )
            )

    def test_zero_new_tokens(self):
        runner = HfStaticRunner(self._load())
        self.assertEqual(runner.generate(self.PROMPT, 0), [])
        self.assertEqual(runner.filled, len(self.PROMPT))


if __name__ == "__main__":
    unittest.main()
