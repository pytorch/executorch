import argparse
import logging
import os

from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.examples.qualcomm.oss_scripts.hf_causal_lm import inference

from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.exporters import ExecutorchExporter, ExecutorchQnnLlmConfig


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

DECODER_MODEL = "llama3_2-1b"
PTE_FILENAME = "hf_causal_lm_qnn"


def build_model(model_id: str, max_seq_len: int):
    """Load the model the way the QNN LLM path needs it.

    `get_qnn_llm_edge_manager_from_model` decorates `model.config` but does not control how the
    model was loaded, so the dtype / attention / cache choices have to match what
    `get_qnn_llm_edge_manager` would have used: fp32 on CPU with eager attention and a static cache.
    """
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="float32",
        attn_implementation="eager",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=max_seq_len,
            cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
        ),
    ).eval()


def main(args):

    import transformers

    print(f"transformers module: {transformers.__file__}")
    assert "site-packages" not in transformers.__file__, (
        "Follow the setup_qnn_hf.md to use the hf-transformers under repo and not the one under "
        "conda site-package."
    )
    print(
        "This is a hardcoded version serving as a PoC, so it will only run Llama3.2 1B"
    )
    logging.info(f"Loading {args.decoder_model_id}...")
    model = build_model(args.decoder_model_id, args.max_seq_len)

    if not args.pre_gen_pte:
        # The whole export is this one call; everything after it is this script's business.
        et_program_manager = ExecutorchExporter().export(
            model,
            None,
            ExecutorchQnnLlmConfig(
                source_model_config=model.config,
                model_id=args.decoder_model_id,
                max_seq_len=args.max_seq_len,
                soc_model=args.soc_model,
                artifact_dir=args.artifact,
                use_fp16=args.use_fp16,
                calibration_dataset=[args.prompt],
                calibration_tasks=args.calibration_tasks,
                calibration_limit=args.calibration_limit,
                quantizer=None if args.use_fp16 else "qnn",
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )

        # `inference` looks for the .pte under this exact name, so keep it in sync with hf_causal_lm.
        pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
        with open(pte_path, "wb") as f:
            et_program_manager.write_to_file(f)
        logging.info(f"wrote {pte_path} ({os.path.getsize(pte_path)} bytes)")

    if not args.compile_only:
        inference(args, QnnConfig.load_config(args), PTE_FILENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="hf_exporter_smoke")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument(
        "--use_fp16", action="store_true", help="Skip PT2E quantization."
    )
    parser.add_argument(
        "--decoder_model_id",
        help="The Hugging Face ID to export, e.g., 'NousResearch/Llama-3.2-1B'",
        default="NousResearch/Llama-3.2-1B",
    )
    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )

    args = parser.parse_args()
    main(args)
