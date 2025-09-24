import argparse
import numpy as np
import os
import random
import subprocess
import torch

from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper


### GLOBALS
qnn_sdk = os.getenv("QNN_SDK_ROOT")
workspace = "/data/local/tmp/et_ga_benchmark"
memory_script_file = "peak_memory.sh"
perf_file = "statistics.txt"
seed = 1126
###


def image_classification_eval(
    backend,
    soc_model,
    device,
    host,
    pte_path,
    module,
    inputs,
    targets,
    artifact_dir,
):
    from executorch.examples.qualcomm.utils import (
        make_output_dir,
        SimpleADB,
        topk_accuracy,
    )
    import numpy as np
    from pathlib import Path

    adb = SimpleADB(
        qnn_sdk=qnn_sdk,
        build_path="build-android",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{Path(pte_path).stem}",
        device_id=device,
        host_id=host,
        soc_model=soc_model,
    )
    files = ["build-xnnpack/executor_runner"] if backend == "xnn" else None
    custom_commands = (
        f"cd {adb.workspace} && ./executor_runner --model_path "
        f"{os.path.basename(adb.pte_path[0])} --input_list_path input_list.txt"
        if backend == "xnn" else None
    )
    adb.push(inputs=inputs, files=files)
    adb.execute(custom_runner_cmd=custom_commands)

    # collect output data
    output_data_folder = f"{artifact_dir}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=artifact_dir)

    # top-k analysis
    predictions, goldens = [], []
    for input in inputs:
        goldens.append(module(*input).logits.detach().numpy())

    for i in range(len(inputs)):
        predictions.append(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
        )

    k_val = [1, 5]
    topk = [topk_accuracy(goldens, targets, k).item() for k in k_val]
    print("cpu:")
    for i, k in enumerate(k_val):
        print(f"top_{k}->{topk[i]}%")
    topk = [topk_accuracy(predictions, targets, k).item() for k in k_val]
    print("device:")
    for i, k in enumerate(k_val):
        print(f"top_{k}->{topk[i]}%")


def masked_lm_eval(
    backend,
    soc_model,
    device,
    host,
    pte_path,
    module,
    inputs,
    targets,
    artifact_dir,
):
    from executorch.examples.qualcomm.utils import SimpleADB, make_output_dir
    import numpy as np
    from pathlib import Path
    import evaluate

    adb = SimpleADB(
        qnn_sdk=qnn_sdk,
        build_path="build-android",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{Path(pte_path).stem}",
        device_id=device,
        host_id=host,
        soc_model=soc_model,
    )
    files = ["build-xnnpack/executor_runner"] if backend == "xnn" else None
    custom_commands = (
        f"cd {adb.workspace} && ./executor_runner --model_path"
        f" {os.path.basename(adb.pte_path[0])} --input_list_path input_list.txt"
        if backend == "xnn" else None
    )
    if backend == "xnn":
        for i, input in enumerate(inputs):
            inputs[i] = tuple(inp.to(torch.long) for inp in input)

    adb.push(inputs=inputs, files=files)
    adb.execute(custom_runner_cmd=custom_commands)

    # collect output data
    output_data_folder = f"{artifact_dir}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=artifact_dir)

    labels, goldens, predictions = [], [], []
    for i in range(len(inputs)):
        indice = [i for i, x in enumerate(targets[i]) if x != -100]
        labels.extend(targets[i][indice].tolist())
        golden = module(*inputs[i]).logits.detach().numpy().argmax(axis=-1)
        goldens.extend(golden[0, indice].tolist())
        prediction = (
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
            .reshape([1, inputs[0][0].shape[1], -1])
            .argmax(axis=-1)
        )
        predictions.extend(prediction[0, indice].tolist())

    metric = evaluate.load("accuracy")
    results = metric.compute(predictions=goldens, references=labels)
    print(f"cpu accuracy: {results['accuracy']}")
    results = metric.compute(predictions=predictions, references=labels)
    print(f"device accuracy: {results['accuracy']}")


def t5_eval(
    backend,
    soc_model,
    device,
    host,
    pte_path,
    module,
    inputs,
    targets,
    artifact_dir,
):
    from executorch.examples.qualcomm.utils import (
        evaluate_squad, make_output_dir, SimpleADB
    )
    from executorch.examples.qualcomm.oss_scripts.t5.t5 import Seq2SeqLMExportableModulePipeline
    from pathlib import Path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    _, _, spiece_model, _, _ = tokenizer.save_pretrained(artifact_dir)
    max_seq_len = module.decoder.max_static_cache_length

    workspace = f"/data/local/tmp/executorch/{Path(pte_path).stem}"
    adb = SimpleADB(
        qnn_sdk=qnn_sdk,
        build_path="build-android",
        pte_path=pte_path,
        workspace=workspace,
        device_id=device,
        host_id=host,
        soc_model=soc_model,
        runner="examples/qualcomm/oss_scripts/t5/qnn_t5_runner",
    )
    runner_args = " ".join(
        [
            f"--tokenizer_model_path {os.path.basename(spiece_model)}",
            f"--model_path {os.path.basename(pte_path)}",
            f"--seq_len {max_seq_len}",
            "--output_folder_path outputs",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./{'qnn_t5_runner' if backend == 'qnn' else 'xnn_t5_runner'}",
            runner_args,
        ]
    )
    files = [spiece_model]
    if backend == "xnn":
        files.append("build-xnnpack/xnn_t5_runner")

    adb.push(inputs=inputs, files=files)
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{artifact_dir}/outputs"
    make_output_dir(output_data_folder)

    outputs = []
    def post_process():
        for i in range(len(inputs)):
            with open(f"{artifact_dir}/outputs/output_{i}.txt", "r") as f:
                outputs.append(f.read())
    adb.pull(output_path=artifact_dir, callback=post_process)

    # cpu inference
    goldens = []
    with torch.no_grad():
        for input in inputs:
            # run encoder
            hidden_state = module.encoder(*input[:-1])
            _, attn_mask, _, _, pos = module.decoder.get_example_inputs()
            tokens = [input[-1].item()]
            # generate tokens one by one
            for _ in range(max_seq_len - 1):
                # run decoder for next token prediction
                logits = module.decoder(
                    torch.tensor([[tokens[-1]]], dtype=torch.long),
                    attn_mask,
                    hidden_state,
                    input[1],
                    pos,
                )

                # get next token
                tokens.append(torch.argmax(logits, dim=-1).item())
                pos += 1
                attn_mask[..., pos] = 0

                # Check if EOS token
                if tokens[-1] == module.decoder.config.eos_token_id:
                    break
            goldens.append(tokenizer.decode(tokens[1:-1]))

    print("cpu accuracy >")
    Seq2SeqLMExportableModulePipeline.evaluate_with_ground_truth(
        tokenizer, goldens, targets, evaluate_squad
    )
    print("device accuracy >")
    Seq2SeqLMExportableModulePipeline.evaluate_with_ground_truth(
        tokenizer, outputs, targets, evaluate_squad
    )


def whisper_eval(
    backend,
    soc_model,
    device,
    host,
    pte_path,
    module,
    inputs,
    targets,
    artifact_dir,
):
    from executorch.examples.qualcomm.utils import make_output_dir, SimpleADB
    from executorch.examples.qualcomm.oss_scripts.whisper.whisper import eval_metric
    from executorch.examples.qualcomm.oss_scripts.whisper.whisper_model import EncoderDecoderCache, DynamicCache
    from pathlib import Path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
    tokenizer_json = tokenizer.save_pretrained(args.artifact)[-1]
    max_seq_len = module.max_seq_length

    workspace = f"/data/local/tmp/executorch/{Path(pte_path).stem}"
    adb = SimpleADB(
        qnn_sdk=qnn_sdk,
        build_path="build-android",
        pte_path=pte_path,
        workspace=workspace,
        device_id=device,
        host_id=host,
        soc_model=soc_model,
        runner="examples/qualcomm/oss_scripts/whisper/qnn_whisper_runner",
    )
    runner_args = " ".join(
        [
            f"--model_path {os.path.basename(pte_path)}",
            f"--tokenizer_json_path {os.path.basename(tokenizer_json)}",
            "--input_list_path input_list.txt",
            f"--seq_len {max_seq_len}",
            "--output_folder_path outputs",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./{'qnn_whisper_runner' if backend == 'qnn' else 'xnn_whisper_runner'}",
            runner_args,
        ]
    )
    files = [tokenizer_json]
    if backend == "xnn":
        files.append("build-xnnpack/xnn_whisper_runner")

    adb.push(inputs=inputs, files=files)
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{artifact_dir}/outputs"
    make_output_dir(output_data_folder)

    outputs = []
    def post_process():
        for i in range(len(inputs)):
            with open(f"{artifact_dir}/outputs/output_{i}.txt", "r") as f:
                outputs.append(f.read())
    adb.pull(output_path=artifact_dir, callback=post_process)

    # cpu inference
    decoder_start_token_id = getattr(module.config, "decoder_start_token_id", 50258)
    eos_token_id = getattr(module.config, "eos_token_id", 50257)
    goldens = []
    with torch.no_grad():
        for input in inputs:
            # run encoder
            hidden_state = module.whisper_encoder(*input)
            _, attn_mask, _, pos = module.whisper_decoder.get_example_inputs()
            tokens = [decoder_start_token_id]
            # generate tokens one by one
            for _ in range(max_seq_len - 1):
                # run decoder for next token prediction
                logits = module.whisper_decoder(
                    torch.tensor([[tokens[-1]]], dtype=torch.long),
                    attn_mask,
                    hidden_state,
                    pos,
                )

                # get next token
                tokens.append(torch.argmax(logits, dim=-1).item())
                pos += 1
                attn_mask[..., pos] = 0

                # Check if EOS token
                if tokens[-1] == eos_token_id:
                    break

            module.whisper_decoder.static_cache.reset()
            module.whisper_decoder.cache = EncoderDecoderCache(
                module.whisper_decoder.static_cache, DynamicCache()
            )
            goldens.append(tokenizer.decode(tokens[1:]))

    print(f"cpu accuracy >\n{eval_metric(goldens, targets)}")
    print(f"device accuracy >\n{eval_metric(outputs, targets)}")


class RunnerEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class to run PPL scores on device.
    """

    def __init__(
        self,
        backend,
        soc_model,
        device,
        host,
        pte_path,
        artifact_dir,
        decoder_model,
        tokenizer,
        runtime_tokenizer_path,
    ):
        from pathlib import Path
        from executorch.exir._serialize._program import deserialize_pte_binary
        from executorch.examples.qualcomm.utils import SimpleADB

        self.pte_path = pte_path
        with open(pte_path, "rb") as f:
            program_data = f.read()
        program = deserialize_pte_binary(program_data)
        # Retrieve vocab_size from get_metadata under static_llama that is passed to edge manager
        self.output_vocab_size = None
        pte_max_seq_len = None
        self.logits_scale = None
        self.logits_zero_point = None
        self.kv_io_bit_width = 32
        self.et_backend = backend
        for method in program.execution_plan:
            # Don't use tokenizer.n_words, the numbers are off once calling get_tokenizer()
            if method.name == "get_vocab_size":
                # pyre-ignore
                self.output_vocab_size = method.values[0].val.int_val
            if method.name == "get_max_seq_len":
                # pyre-ignore
                pte_max_seq_len = method.values[0].val.int_val
            if method.name == "get_logits_scale":
                self.logits_scale = method.values[0].val.double_val
            if method.name == "get_logits_zero_point":
                self.logits_zero_point = method.values[0].val.int_val
            if method.name == "get_kv_io_bit_width":
                self.kv_io_bit_width = method.values[0].val.int_val

        # FP has no scale/zero_point, use following values, which is equivalent to not performing dequantize.
        if self.kv_io_bit_width == 32:
            self.logits_scale = 1
            self.logits_zero_point = 0
        elif self.logits_scale is None or self.logits_zero_point is None:
            raise RuntimeError(
                "Unable to find scale/offset. The .pte file might be deprecated. Please generate a new .pte file"
            )

        assert self.output_vocab_size is not None, "Couldn't find the vocab size"
        assert pte_max_seq_len is not None, "Couldn't find the max_seq_len from pte"
        self.decoder_model = decoder_model
        self.max_seq_length = pte_max_seq_len
        self.runtime_tokenizer_path = runtime_tokenizer_path
        self.artifact_dir = artifact_dir
        self.output_dir = args.artifact
        self.workspace = f"/data/local/tmp/executorch/{decoder_model}"
        self.adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path="build-android",
            pte_path=pte_path,
            workspace=self.workspace,
            device_id=device,
            host_id=host,
            soc_model=soc_model,
            runner="examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
        )
        files = [self.runtime_tokenizer_path]
        if backend == "xnn":
            files.append("build-xnnpack/xnn_llama_runner")
        self.adb.push(inputs=[], files=files)
        # n seq len = n-1 cache len, so we len(inps) = n-1 during _model_call
        # pyre-ignore
        super().__init__(None, tokenizer, self.max_seq_length - 1)

    def _model_call(self, inps):
        from executorch.examples.qualcomm.oss_scripts.llama import DECODER_MODEL_VERSION
        from executorch.examples.qualcomm.utils import make_output_dir

        input_file_name = f"{self.artifact_dir}/input_tokens.raw"
        inps = inps.to(torch.uint64).numpy()
        inps.tofile(input_file_name)

        outputs_path = "outputs/outputs.txt"
        dump_logits_path = "outputs/all_logit.raw"
        performance_output_path = "outputs/inference_speed.txt"
        runner_cmd = " ".join(
            [
                f"cd {self.workspace} &&",
                f"./{self.et_backend}_llama_runner",
                f"--decoder_model_version {DECODER_MODEL_VERSION[self.decoder_model]}",
                f"--tokenizer_path {os.path.basename(self.runtime_tokenizer_path)}",
                f"--model_path {os.path.basename(self.pte_path)}",
                f"--seq_len {self.max_seq_length}",
                f"--output_path {outputs_path}",
                f"--performance_output_path {performance_output_path}",
                f"--kv_updater SmartMask",
                f"--eval_mode 0",
                "--temperature 0",
                f"--dump_logits_path {dump_logits_path}",
                f"--tokenized_prompt {os.path.basename(input_file_name)}",
            ]
        )

        self.adb.push(inputs=[], files=[input_file_name], init_env=False)
        self.adb.execute(custom_runner_cmd=runner_cmd)
        output_data_folder = f"{self.output_dir}/outputs"
        make_output_dir(output_data_folder)
        output_tensor_list = []

        def post_process():
            with open(f"{self.artifact_dir}/{dump_logits_path}", "r") as f:
                if self.kv_io_bit_width == 32:
                    output_tensor = torch.from_numpy(
                        np.fromfile(f.name, dtype=np.float32).reshape(
                            1, -1, self.output_vocab_size
                        )
                    )
                    output_tensor_list.append(output_tensor)
                else:
                    output_tensor = torch.from_numpy(
                        np.fromfile(f.name, dtype=np.uint16).reshape(
                            1, -1, self.output_vocab_size
                        )
                    )
                    output_tensor = (
                        output_tensor.to(torch.float32) - self.logits_zero_point
                    ) * self.logits_scale
                    output_tensor_list.append(output_tensor)

            # simple_eval will run multiple rounds, use last run for inference speed
            with open(f"{self.artifact_dir}/{performance_output_path}", "r") as f:
                self.inference_speed = float(f.read())

        self.adb.pull(output_path=self.output_dir, callback=post_process)
        return output_tensor_list[0]


def llm_eval(
    backend,
    soc_model,
    device,
    host,
    pte_path,
    module,
    decoder_model,
    decoder_model_config,
    artifact_dir,
    **kwargs,
):
    import json
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
        GraphModuleCalibrationWrapper, smart_mask_updater
    )
    from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
    from transformers import AutoTokenizer
    try:
        from lm_eval.evaluator import simple_evaluate
    except ImportError:
        raise ImportError(
            "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
        )

    # Tokenizer related
    if "llama3_2" in decoder_model:
        tokenizer = get_tokenizer(kwargs["tokenizer_model"])
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), f"Wrong tokenizer provided for llama3_2."
        runtime_tokenizer_path = args.tokenizer_model
    else:
        model_id = decoder_model_config.repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_artifacts = tokenizer.save_pretrained(artifact_dir)
        tokenizer_config = tokenizer_artifacts[0]
        runtime_tokenizer_path = tokenizer_artifacts[-1]
        tokenizer = get_tokenizer(runtime_tokenizer_path, tokenizer_config)

    if decoder_model == "phi_4_mini":
        with open(runtime_tokenizer_path, "r+") as file:
            data = json.load(file)
            data["pre_tokenizer"]["pretokenizers"][-2]["invert"] = False
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

    # on device
    # Generate the eval wrapper
    device_eval_wrapper = RunnerEvalWrapper(
        backend=backend,
        soc_model=soc_model,
        device=device,
        host=host,
        pte_path=pte_path,
        artifact_dir=artifact_dir,
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        runtime_tokenizer_path=runtime_tokenizer_path,
    )
    # Evaluate the model on device
    with torch.no_grad():
        device_eval_results = simple_evaluate(
            model=device_eval_wrapper,
            tasks=["wikitext"],
            num_fewshot=None,
            limit=1,
        )

    # on host
    # Generate the eval wrapper
    cpu_eval_wrapper = GraphModuleCalibrationWrapper(
        model=module,
        tokenizer=tokenizer,
        max_seq_length=1024,
        ar_len=1,
        use_kv_cache=True,
        get_example_inputs=module.get_example_inputs,
        kv_updater=smart_mask_updater,
        use_i64_token=False,
        seq_mse_candidates=0,
    )
    # Evaluate the model on device
    with torch.no_grad():
        cpu_eval_results = simple_evaluate(
            model=cpu_eval_wrapper,
            tasks=["wikitext"],
            num_fewshot=None,
            limit=1,
        )

    print("cpu accuracy >")
    print(cpu_eval_results["results"]["wikitext"]["word_perplexity,none"])
    print("device accuracy >")
    print(device_eval_results["results"]["wikitext"]["word_perplexity,none"])


def get_model_dispatcher(dataset_path, **kwargs):
    from transformers import AutoModelForMaskedLM, AutoModelForImageClassification
    from executorch.examples.qualcomm.utils import (
        get_imagenet_dataset, get_masked_language_model_dataset
    )

    def get_masked_lm_sample_input(pretrained, data_size=100):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        return get_masked_language_model_dataset(dataset_path, tokenizer, data_size)

    def get_albert():
        pretrained = "albert/albert-base-v2"
        inputs, targets = get_masked_lm_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, masked_lm_eval

    def get_bert():
        pretrained = "google-bert/bert-base-uncased"
        inputs, targets = get_masked_lm_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, masked_lm_eval

    def get_cvt():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/cvt-13").eval()
        return module, inputs, targets, image_classification_eval

    def get_deit():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224").eval()
        return module, inputs, targets, image_classification_eval

    def get_efficientnet():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0").eval()
        return module, inputs, targets, image_classification_eval

    def get_eurobert():
        pretrained = "EuroBERT/EuroBERT-210m"
        inputs, targets = get_masked_lm_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained, trust_remote_code=True).eval()
        return module, inputs, targets, masked_lm_eval

    def get_distilbert():
        pretrained = "distilbert/distilbert-base-uncased"
        inputs, targets = get_masked_lm_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, masked_lm_eval

    def get_dit():
        from executorch.examples.qualcomm.oss_scripts.dit import get_rvlcdip_dataset
        inputs, targets = get_rvlcdip_dataset(100)
        module = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip").eval()
        return module, inputs, targets, image_classification_eval

    def get_focalnet():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/focalnet-tiny").eval()
        return module, inputs, targets, image_classification_eval

    def get_mobilevit_v1():
        import executorch.examples.qualcomm.oss_scripts.mobilevit_v1 as mvit1
        inputs, targets = mvit1.get_imagenet_dataset(dataset_path, 100)
        module = AutoModelForImageClassification.from_pretrained("apple/mobilevit-xx-small").eval()
        return module, inputs, targets, image_classification_eval

    def get_mobilevit_v2():
        import executorch.examples.qualcomm.oss_scripts.mobilevit_v2 as mvit2
        inputs, targets = mvit2.get_imagenet_dataset(dataset_path, 100)
        module = AutoModelForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").eval()
        return module, inputs, targets, image_classification_eval

    def get_pvt():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("Zetatech/pvt-tiny-224").eval()
        return module, inputs, targets, image_classification_eval

    def get_roberta():
        pretrained = "xlm-roberta-base"
        inputs, targets = get_masked_lm_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, masked_lm_eval

    def get_swin():
        inputs, targets = get_imagenet_dataset(dataset_path, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
        return module, inputs, targets, image_classification_eval

    def get_t5():
        from executorch.examples.qualcomm.utils import get_seq2seq_dataset_from_squad_csv
        from executorch.examples.qualcomm.oss_scripts.t5.t5 import T5
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
        max_hidden_seq_length = 384
        max_cache_length = 512
        module = T5(
            model,
            tokenizer,
            max_hidden_seq_length=max_hidden_seq_length,
            max_cache_length=max_cache_length,
        )
        inputs, targets = get_seq2seq_dataset_from_squad_csv(
            args.dataset,
            tokenizer,
            100,
            max_hidden_seq_length=max_hidden_seq_length,
        )
        return module, inputs, targets, t5_eval

    def get_whisper():
        from executorch.examples.qualcomm.oss_scripts.whisper.whisper import (
            get_dataset, Whisper
        )
        from transformers import AutoModelForSpeechSeq2Seq
        module = (
            AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
            .to("cpu")
            .eval()
        )
        max_cache_length = 1024
        max_seq_length = 1024
        batch_size = 1
        module = Whisper(
            module,
            batch_size=batch_size,
            max_cache_length=max_cache_length,
            max_seq_length=max_seq_length,
        )
        inputs, targets = get_dataset(100)
        return module, inputs, targets, whisper_eval

    def get_static_llama(decoder_model, decoder_model_config, **kwargs):
        import json
        from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
        from executorch.examples.qualcomm.oss_scripts.llama.llama import (
            download_and_convert_hf_checkpoint,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
            LlamaModel,
            ModelArgs,
        )
        if "params" in kwargs:
            params_path = kwargs["params"]
        else:
            params_path = decoder_model_config.params_path
        with open(params_path) as f:
            kv_config = ModelArgs(**json.load(f))

        kv_config.max_batch_size = 1
        kv_config.max_seq_len = 1024
        kv_config.use_kv_cache = True
        kv_config.enable_r3 = decoder_model_config.r3
        kv_config.kv_io_bit_width = decoder_model_config.get_kv_io_bit_width()
        kv_config.enable_masked_softmax = decoder_model_config.masked_softmax

        extra_kwargs = {}
        if decoder_model == "gemma3-1b":
            from transformers import Gemma3Config

            hf_config = Gemma3Config.from_pretrained(decoder_model_config.repo_id)
            extra_kwargs["layer_types"] = hf_config.text_config.layer_types
            extra_kwargs["rope_local_base_freq"] = (
                hf_config.text_config.rope_local_base_freq
            )
            extra_kwargs["sliding_window"] = hf_config.sliding_window

        with torch.device("meta"):
            llama_instance = LLM_VARIANT_ARCHS.get(
                decoder_model, LlamaModel)(
                    kv_config,
                    ar_len=1,
                    output_new_cache_only=True,
                    output_cache=True,
                    use_i64_token=False,
                    **extra_kwargs,
                )
        if "checkpoint" not in kwargs:  # HF models
            checkpoint = download_and_convert_hf_checkpoint(
                decoder_model_config.repo_id,
                decoder_model_config.convert_weights.__func__,
            )
            state_dict = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )
            if decoder_model == "gemma3-1b":
                for k, v in state_dict.items():
                    if "norm" not in k:
                        continue
                    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                    # See https://github.com/huggingface/transformers/pull/29402
                    state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
        else:
            state_dict = torch.load(
                kwargs["checkpoint"], weights_only=True, map_location="cpu", mmap=True
            )

        if decoder_model_config.transform_weight:
            # Change to HuggingFace weight to improve the performance of RoPE in HTP backend.
            def permute(w, heads):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                return (
                    w.view(heads, dim_0 // heads // 2, 2, dim_1)
                    .transpose(1, 2)
                    .reshape(dim_0, dim_1)
                )

            for layer_i in range(llama_instance.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"], llama_instance.n_heads
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"], llama_instance.n_kv_heads
                )

        llama_instance.load_state_dict(state_dict, strict=True, assign=True)
        for layer in llama_instance.layers:
            if getattr(layer.attention, "prepare_sha", None):
                layer.attention.prepare_sha()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

        return llama_instance.to(torch.float32)

    def get_decoder_model(model_name, **kwargs):
        from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
        decoder_model_config = SUPPORTED_LLM_MODELS[model_name]
        llama_instance = get_static_llama(model_name, decoder_model_config, **kwargs)
        return llama_instance, model_name, decoder_model_config, llm_eval

    def get_qwen2_5_0_5b():
        return get_decoder_model("qwen2_5-0_5b")

    def get_qwen2_5_1_5b():
        return get_decoder_model("qwen2_5-1_5b")

    def get_qwen3_0_6b():
        return get_decoder_model("qwen3-0_6b")

    def get_qwen3_1_7b():
        return get_decoder_model("qwen3-1_7b")

    def get_smollm2_135m():
        return get_decoder_model("smollm2_135m")

    def get_smollm3_3b():
        return get_decoder_model("smollm3-3b")

    def get_phi_4_mini():
        return get_decoder_model("phi_4_mini")

    def get_llama3_2_1b_instruct(params, tokenizer_model, checkpoint):
        return get_decoder_model(
            "llama3_2-1b_instruct",
            params=params,
            tokenizer_model=tokenizer_model,
            checkpoint=checkpoint
        )

    def get_llama3_2_3b_instruct(params, tokenizer_model, checkpoint):
        return get_decoder_model(
            "llama3_2-3b_instruct",
            params=params,
            tokenizer_model=tokenizer_model,
            checkpoint=checkpoint
        )

    def get_gemma3_1b():
        return get_decoder_model("gemma3-1b")

    model_dict = {
        "albert": get_albert,
        "bert": get_bert,
        "cvt": get_cvt,
        "deit": get_deit,
        "dit": get_dit,
        "distilbert": get_distilbert,
        "efficientnet": get_efficientnet,
        "eurobert": get_eurobert,
        "focalnet": get_focalnet,
        "gemma3-1b": get_gemma3_1b,
        "llama3_2-1b_instruct": get_llama3_2_1b_instruct,
        "llama3_2-3b_instruct": get_llama3_2_3b_instruct,
        "mobilevit_v1": get_mobilevit_v1,
        "mobilevit_v2": get_mobilevit_v2,
        "phi_4_mini": get_phi_4_mini,
        "pvt": get_pvt,
        "qwen2_5-0_5b": get_qwen2_5_0_5b,
        "qwen2_5-1_5b": get_qwen2_5_1_5b,
        "qwen3-0_6b": get_qwen3_0_6b,
        "qwen3-1_7b": get_qwen3_1_7b,
        "roberta": get_roberta,
        "smollm2_135m": get_smollm2_135m,
        "smollm3-3b": get_smollm3_3b,
        "swin": get_swin,
        "t5": get_t5,
        "whisper": get_whisper,
    }
    return model_dict


def get_artifacts(backend, pte_path, soc_model, target_model, **kwargs):
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        DECODER_MODEL_VERSION,
    )
    from executorch.backends.qualcomm.utils.utils import get_soc_to_arch_map

    htp_arch = get_soc_to_arch_map()[soc_model]

    def get_build_dir(backend):
        build_dir = {
            "qnn": "build-android",
            "xnn": "build-xnnpack",
        }
        return build_dir[backend]

    memory_script = """$@ 2> /dev/null &
PROCESS=$(echo $1 | sed -e 's/^\.\///g')
PEAK_MEM=0
SAMPLES=0
TOTAL=0
while true; do
    PID=$(pidof $PROCESS)
    if [ "$PID" != "" ]; then
        DMA=$(dmabuf_dump $PID | grep "PROCESS TOTAL" | awk '{ print $3 }')
        PSS=$(dumpsys meminfo -s $PID | grep "TOTAL PSS" | awk '{ print $3 }')
        if [ "$PSS" == "" ]; then
            continue
        fi
        CURRENT=$(($DMA+$PSS))
        if [ CURRENT -gt PEAK_MEM ]; then
            PEAK_MEM=$CURRENT
        fi
        SAMPLES=$(awk -v s="$SAMPLES" 'BEGIN { print s + 1 }')
        TOTAL=$(awk -v t="$TOTAL" -v c="$CURRENT" 'BEGIN { print t + c }')
    else
        break
    fi
done
echo "peak_mem: $PEAK_MEM" >> statistics.txt
AVG_MEM=$(awk -v total="$TOTAL" -v samples="$SAMPLES" 'BEGIN { printf "%.3f", total / samples }')
echo "avg_mem: $AVG_MEM" >> statistics.txt
    """
    with open(memory_script_file, "w") as f:
        f.write(memory_script)

    runner = {
        "qnn": f"{get_build_dir(backend)}/examples/qualcomm/executor_runner/qnn_executor_runner",
        "xnn": f"{get_build_dir(backend)}/executor_runner",
    }

    artifacts = {
        "qnn": [
            pte_path,
            f"{qnn_sdk}/lib/aarch64-android/libQnnHtp.so",
            (
                f"{qnn_sdk}/lib/hexagon-v{htp_arch}/"
                f"unsigned/libQnnHtpV{htp_arch}Skel.so"
            ),
            (f"{qnn_sdk}/lib/aarch64-android/" f"libQnnHtpV{htp_arch}Stub.so"),
            f"{qnn_sdk}/lib/aarch64-android/libQnnHtpPrepare.so",
            f"{qnn_sdk}/lib/aarch64-android/libQnnSystem.so",
            f"{get_build_dir(backend)}/backends/qualcomm/libqnn_executorch_backend.so",
            f"{qnn_sdk}/lib/aarch64-android/libQnnModelDlc.so",
            runner[backend],
            memory_script_file,
        ],
        "xnn": [
            pte_path,
            runner[backend],
            memory_script_file,
        ],
    }

    if target_model in DECODER_MODEL_VERSION:
        llm_tokenizer = kwargs.get("tokenizer_model", f"{os.path.dirname(pte_path)}/tokenizer.json")
        if backend == "qnn":
            artifacts[backend].append(f"{get_build_dir(backend)}/examples/qualcomm/oss_scripts/llama/{backend}_llama_runner")
        elif backend == "xnn":
            artifacts[backend].append(f"{get_build_dir(backend)}/{backend}_llama_runner")
        artifacts[backend].append(llm_tokenizer)

    return artifacts[backend]


def get_llm_cmds(backend, pte_path, decoder_model, **kwargs):
    common_cmd_args = [
        f"--model_path {os.path.basename(pte_path)}",
        "--seq_len 1024",
        f"--decoder_model_version {decoder_model}",
        f"--tokenizer_path {'tokenizer.model' if 'tokenizer_model' in kwargs else 'tokenizer.json'}",
        "--prompt 'I would like to learn python, could you teach me with a simple example?'",
    ]
    for k, v in kwargs.items():
        common_cmd_args.append(f"{k} {v}")

    cmds_for_inference = (
        " ".join(
            [
                f"cd {workspace} &&",
                f"./{backend}_llama_runner {' '.join(common_cmd_args)}",
            ]
        )
    )

    if backend == "xnn":
        common_cmd_args[1] = "--seq_len 100"
    cmds_for_memory = (
        " ".join(
            [
                f"cd {workspace} &&",
                f"chmod +x {memory_script_file} &&",
                f"./{memory_script_file} ./{backend}_llama_runner {' '.join(common_cmd_args)}",
            ]
        )
    )
    return [cmds_for_inference, cmds_for_memory]


def get_cmds(backend, pte_path, iteration, method_index, target_model, **kwargs):
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        DECODER_MODEL_VERSION,
    )

    if target_model in DECODER_MODEL_VERSION:
        return get_llm_cmds(
            backend, pte_path, DECODER_MODEL_VERSION[target_model], **kwargs
        )

    cmd_args = {
        "qnn": (
            [
                f"--model_path {os.path.basename(pte_path)}",
                f"--iteration {iteration}",
                f"--method_index {method_index}",
                "--dump_statistics",
            ]
        ),
        "xnn": (
            [
                f"--model_path {os.path.basename(pte_path)}",
                f"--num_executions {iteration}",
                f"--method_index {method_index}",
                "--dump_statistics",
            ]
        ),
    }
    cmds_for_inference = {
        "qnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./qnn_executor_runner &&",
                    f"./qnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
        "xnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./executor_runner &&",
                    f"./executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
    }
    # do not dump inference metrics during profiling memory
    for _, v in cmd_args.items():
        v.pop()
    cmds_for_memory = {
        "qnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./qnn_executor_runner &&",
                    f"chmod +x {memory_script_file} &&",
                    f"./{memory_script_file} ./qnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
        "xnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./executor_runner &&",
                    f"chmod +x {memory_script_file} &&",
                    f"./{memory_script_file} ./executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
    }
    return [cmds_for_inference[backend], cmds_for_memory[backend]]


def start_benchmark(artifacts, cmds, device, host):
    import tempfile

    def adb(action):
        if not host:
            actions = ["adb", "-s", device]
        else:
            actions = ["adb", "-H", host, "-s", device]
        actions.extend(action)
        subprocess.run(actions, stdout=subprocess.DEVNULL)

    def post_process():
        subprocess.run(["rm", "-rf", perf_file], stdout=subprocess.DEVNULL)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for file_name in [perf_file]:
                adb(["pull", f"{workspace}/{file_name}", tmp_dir])
                with open(f"{tmp_dir}/{file_name}", "r") as f:
                    print(f.read())

    adb(["shell", "rm", "-rf", workspace])
    adb(["shell", "mkdir", "-p", workspace])
    for artifact in artifacts:
        adb(["push", artifact, workspace])
    for cmd in cmds:
        adb(["shell", cmd])
    post_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--backend",
        help="either 'qnn' or 'xnn'",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pte",
        help="path to .pte",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="path to generated intermediate artifacts",
    )
    parser.add_argument(
        "-t",
        "--target_model",
        help=f"supported targets: {get_model_dispatcher('').keys()}",
        required=True,
    )
    parser.add_argument(
        "-H",
        "--host",
        help="hostname for adb gateway",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for adb device",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--soc_model",
        help="model name of SoC",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--iteration",
        help="total number of inferences",
        default=100,
    )
    parser.add_argument(
        "-e",
        "--eval",
        help="perform e2e evaluation for checking accuracy metrics",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="specify dataset path for evaluation",
    )
    parser.add_argument(
        "--method_index",
        help="specify which method to be executed",
        default=0,
    )
    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kwargs = {}
    if all([args.params, args.tokenizer_model, args.checkpoint]):
        kwargs = {
            "params": args.params,
            "tokenizer_model": args.tokenizer_model,
            "checkpoint": args.checkpoint,
        }

    if args.eval:
        module, inputs, targets, eval_func = get_model_dispatcher(
            args.dataset, **kwargs
        )[args.target_model](**kwargs)
        eval_func(
            args.backend,
            args.soc_model,
            args.device,
            args.host,
            args.pte,
            module,
            inputs,
            targets,
            args.artifact,
            **kwargs,
        )
    else:
        start_benchmark(
            artifacts=get_artifacts(
                args.backend, args.pte, args.soc_model, args.target_model, **kwargs
            ),
            cmds=get_cmds(
                args.backend,
                args.pte,
                args.iteration,
                args.method_index,
                args.target_model,
                **kwargs,
            ),
            device=args.device,
            host=args.host,
        )
