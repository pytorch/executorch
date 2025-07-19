import argparse
import logging
from typing import Dict

import coremltools as ct

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)

from optimum.exporters.executorch.recipe_registry import register_recipe

from tabulate import tabulate
from torch.export import ExportedProgram

COREML_RECIPE = "coreml-et-testing"
COREML_RECIPE_LLM_KWARGS = {
    # FP16 does not give good numerics for some LLMs
    "compute_precision": ct.precision.FLOAT32,
    "quant_recipe": "4bit",
    "compute_unit": ct.ComputeUnit.CPU_AND_GPU,
    "minimum_deployment_target": ct.target.iOS18,
}
COREML_RECIPE_FP32_KWARGS = {
    "compute_precision": ct.precision.FLOAT32,
    "compute_unit": ct.ComputeUnit.CPU_AND_GPU,
}
COREML_RECIPE_FP16_KWARGS = {
    "compute_precision": ct.precision.FLOAT16,
    "compute_unit": ct.ComputeUnit.ALL,
}


@register_recipe(COREML_RECIPE)
def export_to_executorch_with_coreml(
    model,
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to CoreML backend.

    This function also write metadata required by the ExecuTorch runtime to the model.

    Args:
        model (Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule]):
            The PyTorch model to be exported to ExecuTorch.
        **kwargs:
            Additional keyword arguments for recipe-specific configurations, e.g. export using different example inputs, or different compile/bechend configs.

    Returns:
        Dict[str, ExecutorchProgram]:
            A map of exported and optimized program for ExecuTorch.
            For encoder-decoder models or multimodal models, it may generate multiple programs.
    """
    # Import here because coremltools might not be available in all environments
    import coremltools as ct

    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
        **kwargs,
    ) -> Dict[str, ExecutorchProgram]:
        valid_kwargs = [
            "compute_unit",
            "minimum_deployment_target",
            "compute_precision",
            "model_type",
            "take_over_mutable_buffer",
            "quant_recipe",
        ]
        for k in kwargs:
            if k not in valid_kwargs:
                raise ValueError(
                    f"Invalid keyword argument {k} for CoreML recipe. Valid arguments are {valid_kwargs}"
                )

        compute_unit = kwargs.get("compute_unit", ct.ComputeUnit.ALL)
        minimum_deployment_target = kwargs.get(
            "minimum_deployment_target", ct.target.iOS15
        )
        compute_precision = kwargs.get("compute_precision", ct.precision.FLOAT16)
        model_type = kwargs.get("model_type", "model")
        model_type = {
            "model": CoreMLBackend.MODEL_TYPE.MODEL,
            "modelc": CoreMLBackend.MODEL_TYPE.COMPILED_MODEL,
        }[model_type]
        take_over_mutable_buffer = kwargs.get(
            "take_over_mutable_buffer", (minimum_deployment_target >= ct.target.iOS18)
        )

        op_linear_quantizer_config = None
        quant_recipe = kwargs.get("quant_recipe", None)
        valid_quant_recipes = {
            "8bit": {
                "mode": "linear_symmetric",
                "dtype": "int8",
                "granularity": "per_channel",
            },
            "4bit": {
                "mode": "linear_symmetric",
                "dtype": "int4",
                "granularity": "per_block",
                "block_size": 32,
            },
        }
        if quant_recipe is not None and quant_recipe not in valid_quant_recipes:
            raise ValueError(
                f"Invalid quant recipe {quant_recipe}, must be one of {valid_quant_recipes.keys()}"
            )
        op_linear_quantizer_config = valid_quant_recipes.get(quant_recipe, None)

        et_progs = {}
        backend_config_dict = {}
        for pte_name, exported_program in exported_programs.items():
            exported_program = exported_program.run_decompositions({})
            logging.debug(f"\nExported program for {pte_name}.pte: {exported_program}")
            et_progs[pte_name] = to_edge_transform_and_lower(
                exported_program,
                partitioner=[
                    CoreMLPartitioner(
                        # Do not delegate embedding because it leads to a compression conflict
                        skip_ops_for_coreml_delegation=[
                            "aten.embedding.default",
                        ],
                        compile_specs=CoreMLBackend.generate_compile_specs(
                            compute_unit=compute_unit,
                            minimum_deployment_target=minimum_deployment_target,
                            compute_precision=compute_precision,
                            model_type=model_type,
                            op_linear_quantizer_config=op_linear_quantizer_config,
                        ),
                        take_over_mutable_buffer=take_over_mutable_buffer,
                    )
                ],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    # In ET 0.7, we can set _skip_dim_order=False
                    _skip_dim_order=True,
                ),
                constant_methods=metadata,
            ).to_executorch(
                config=ExecutorchBackendConfig(**backend_config_dict),
            )
            logging.debug(
                f"\nExecuTorch program for {pte_name}.pte: {et_progs[pte_name].exported_program().graph_module}"
            )
            delegation_info = get_delegation_info(
                et_progs[pte_name].exported_program().graph_module
            )
            logging.debug(
                f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}"
            )
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return et_progs

    exported_progs = model.export()
    return _lower_to_executorch(exported_progs, model.metadata, **kwargs)


# model_id = "HuggingFaceTB/SmolLM2-135M" # works, output questionable
# model_id = "NousResearch/Llama-3.2-1B" # works at 4-bit (output questionable)
# model_id = "microsoft/Phi-4-mini-instruct" # fails export
# model_id = "Qwen/Qwen3-0.6B" # works at 4-bit (nonsense output)
# model_id = "allenai/OLMo-1B-hf" # works at 4-bit (bad output)
def test_decoder_only_model(model_id):
    from optimum.executorch import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer

    prompt = "Simply put, the theory of relativity states"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = ExecuTorchModelForCausalLM.from_pretrained(
        model_id,
        recipe=COREML_RECIPE,
        recipe_kwargs=COREML_RECIPE_LLM_KWARGS,
    )

    generated_text = model.text_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        max_seq_len=64,
    )
    print(f"\nGenerated text:\n\t{generated_text}")


# All models fail export with same issue as https://fb.workplace.com/groups/pytorch.edge.users/permalink/1796069037930048/
# Even if fixed, bypassed they fail lowering to CoreML because symints are passed to signature
# model_id = "google-bert/bert-base-uncased"
# model_id = "distilbert/distilbert-base-uncased"
# model_id = "FacebookAI/xlm-roberta-base"
def test_encoder_only_model(model_id):
    from optimum.executorch import ExecuTorchModelForMaskedLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ExecuTorchModelForMaskedLM.from_pretrained(
        model_id=model_id,
        recipe=COREML_RECIPE,
        recipe_kwargs=COREML_RECIPE_LLM_KWARGS,
    )

    input_text = f"Paris is the {tokenizer.mask_token} of France."
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        max_length=10,
    )

    # Test inference using ExecuTorch model
    exported_outputs = model.forward(inputs["input_ids"], inputs["attention_mask"])
    predicted_masks = tokenizer.decode(exported_outputs[0, 4].topk(5).indices)
    print(f"Predicted masks: {predicted_masks}")
    assert any(
        word in predicted_masks for word in ["capital", "center", "heart", "birthplace"]
    )


# model_id = "openai/whisper-tiny" works
def test_whisper(model_id):
    from datasets import load_dataset
    from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
    from transformers import AutoProcessor, AutoTokenizer

    model_id = "openai/whisper-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ExecuTorchModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        recipe=COREML_RECIPE,
        recipe_kwargs=COREML_RECIPE_LLM_KWARGS,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    dataset = load_dataset(
        "distil-whisper/librispeech_long", "clean", split="validation"
    )
    sample = dataset[0]["audio"]

    input_features = processor(
        sample["array"],
        return_tensors="pt",
        truncation=False,
        sampling_rate=sample["sampling_rate"],
    ).input_features

    # Current implementation of the transcibe method accepts up to 30 seconds of audio, therefore I trim the audio here.
    input_features_trimmed = input_features[:, :, :3000].contiguous()

    generated_transcription = model.transcribe(tokenizer, input_features_trimmed)
    expected_text = " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really Greek after all, and can discover that."
    print(f"Generated transcription: {generated_transcription}")
    print(f"Expected transcription: {expected_text}")


# model_id = "google-t5/t5-small"
def test_t5(model_id):
    from optimum.executorch import ExecuTorchModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ExecuTorchModelForSeq2SeqLM.from_pretrained(
        model_id, recipe=COREML_RECIPE, recipe_kwargs=COREML_RECIPE_LLM_KWARGS
    )

    article = (
        " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
        " year later, she got married again in Westchester County, but to a different man and without divorcing"
        " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
        ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
        " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
        ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
        ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
        " license application, according to court documents. Prosecutors said the marriages were part of an"
        " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
        " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
        " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
        " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
        " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
        " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
        " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
        " said the immigration scam involved some of her husbands, who filed for permanent residence status"
        " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
        " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
        " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
        ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
        " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
        " native Pakistan after an investigation by the Joint Terrorism Task Force."
    )
    article = "summarize: " + article.strip()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generated_text = model.text_generation(
        tokenizer=tokenizer,
        prompt=article,
    )
    expected_text = 'a year later, she got married again in westchester county, new york. she was married to a different man, but only 18 days after that marriage. she is facing two criminal counts of "offering a false instrument"'
    print(f"Generated text:\n\t{generated_text}")
    print(f"Expected text:\n\t{expected_text}")


# model_id = "google/vit-base-patch16-224"
def test_vit(model_id):
    import torch
    from optimum.executorch import ExecuTorchModelForImageClassification
    from transformers import AutoConfig, AutoModelForImageClassification

    config = AutoConfig.from_pretrained(model_id)
    batch_size = 1
    num_channels = config.num_channels
    height = config.image_size
    width = config.image_size
    pixel_values = torch.rand(batch_size, num_channels, height, width)

    et_model = ExecuTorchModelForImageClassification.from_pretrained(
        model_id=model_id,
        recipe=COREML_RECIPE,
        recipe_kwargs=COREML_RECIPE_FP32_KWARGS,
    )

    eager_model = (
        AutoModelForImageClassification.from_pretrained(model_id).eval().to("cpu")
    )
    with torch.no_grad():
        eager_output = eager_model(pixel_values)
        et_output = et_model.forward(pixel_values)
        assert torch.allclose(eager_output.logits, et_output[0])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
    )
    model_name = parser.parse_args().model_name

    # Decoder-only models
    if model_name == "smollm":
        # Generated text: Simply put, the theory of relativity states that the laws of physics are the same in all places. The theory of relativity is the most important theory in the field of physics. It is the foundation for the rest. The theory of relativity is the foundation for the rest.
        # Decode: 27 tps
        test_decoder_only_model(model_id="HuggingFaceTB/SmolLM2-135M")
    elif model_name == "llama3":
        # Generated text: Simply put, the theory of relativity states that the speed of light is the same for all observers, regardless of their relative motion. This means that the speed of light is the same for an observer moving with respect to the source of light as it is for an observer moving with respect to the observer. This is
        # Decode: 25 tps
        test_decoder_only_model(model_id="NousResearch/Llama-3.2-1B")
    elif model_name == "phi4":
        # Fails export
        # RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: EDGE_DO_NOT_DECOMP::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]. We only support functionalizing operators whose outputs do not have alias annotations (e.g. 'Tensor(a)' is a Tensor with an alias annotation whereas 'Tensor' is a Tensor without. The '(a)' is the alias annotation). The alias annotation specifies that the output Tensor shares storage with an input that has the same annotation. Please check if (1) the output needs to be an output (if not, don't return it), (2) if the output doesn't share storage with any inputs, then delete the alias annotation. (3) if the output indeed shares storage with an input, then add a .clone() before returning it to prevent storage sharing and then delete the alias annotation. Otherwise, please file an issue on GitHub.
        test_decoder_only_model(model_id="microsoft/Phi-4-mini-instruct")
    elif model_name == "qwen3":
        # Generated text: Simply put, the theory of relativity states that the speed of light is constant in all inertial frames of reference. This is a statement that is true, but it does not imply that the speed of light is constant in all inertial frames of reference. The statement that the speed of light is constant in all inert
        # Decode: 20 tps
        test_decoder_only_model(model_id="Qwen/Qwen3-0.6B")
    elif model_name == "olmo":
        # Generated text: Simply put, the theory of relativity states that time and space are not the same. Speed and distance are not the same. Time and distance are not the same. The theory of relativity is a theory of everything. It is a theory of space and time. It is a theory of gravity. It is a
        test_decoder_only_model(model_id="allenai/OLMo-1B-hf")
    elif model_name == "gemma3":
        # Fails export
        # torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <built-in function scaled_dot_product_attention>(*(FakeTensor(..., size=(1, 4, 1, 256)), FakeTensor(..., size=(1, 4, 2048, 256)), FakeTensor(..., size=(1, 4, 2048, 256))), **{'attn_mask': FakeTensor(..., size=(1, 1, 1, 512)), 'dropout_p': 0.0, 'scale': 0.0625, 'is_causal': False}): got RuntimeError('Attempting to broadcast a dimension of length 512 at -1! Mismatching argument at index 1 had torch.Size([1, 1, 1, 512]); but expected shape should be broadcastable to [1, 4, 1, 2048]')
        # From user code:
        #     attn_output = torch.nn.functional.scaled_dot_product_attention(
        test_decoder_only_model(model_id="unsloth/gemma-3-1b-it")
    # Encoder-only models
    elif model_name == "bert":
        # Fails export; same issue as https://fb.workplace.com/groups/pytorch.edge.users/permalink/1796069037930048/
        test_encoder_only_model(model_id="google-bert/bert-base-uncased")
    elif model_name == "distilbert":
        # Fails export; same issue as https://fb.workplace.com/groups/pytorch.edge.users/permalink/1796069037930048/
        test_encoder_only_model(model_id="distilbert/distilbert-base-uncased")
    elif model_name == "roberta":
        # Fails export; same issue as https://fb.workplace.com/groups/pytorch.edge.users/permalink/1796069037930048/
        test_encoder_only_model(model_id="FacebookAI/xlm-roberta-base")
    # Vision models
    elif model_name == "vit":
        # Works
        test_vit(model_id="google/vit-base-patch16-224")
    # Speech models
    elif model_name == "whisper":
        # Works
        # Generated transcription (cut short from expected):
        # Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his Gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and a roast beef, looming before us,
        test_whisper(model_id="openai/whisper-tiny")
    # Seq2Seq models
    elif model_name == "t5":
        # Fails CoreML lowering: NotImplementedError: Placeholder val must be a tensor or fake tensor, but got type <class 'torch.SymInt'>, value s70
        # ET might introduce symbolic variables in graph breaks
        test_t5(model_id="google-t5/t5-small")
    else:
        raise ValueError(f"Invalid model name {model_name}")


if __name__ == "__main__":
    main()
