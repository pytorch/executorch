import os, subprocess, json, torch, types


def mvit_get_imagenet_dataset(dataset_path, data_size):
    from torchvision import datasets
    from transformers import MobileViTFeatureExtractor
    from PIL import Image

    def get_data_loader():
        imagenet_data = datasets.ImageFolder(dataset_path)
        torch.manual_seed(3407)
        return torch.utils.data.DataLoader(imagenet_data, shuffle=True)

    # prepare input data
    inputs, targets, input_list = [], [], ""
    data_loader = get_data_loader()
    feature_extractor = MobileViTFeatureExtractor.from_pretrained(
        "apple/mobilevit-xx-small"
    )
    for index, data in enumerate(data_loader.dataset.imgs):
        if index >= data_size:
            break
        data_path, target = data
        image = Image.open(data_path).convert("RGB")
        feature = feature_extractor(images=image, return_tensors="pt")
        inputs.append((feature["pixel_values"],))
        targets.append(torch.tensor(target))
        input_list += f"input_{index}_0.raw\n"

    return inputs, targets, input_list


def image_eval(module, pte_path, inputs, input_list, targets, artifact_dir, xnn=False):
    from executorch.examples.qualcomm.utils import SimpleADB, make_output_dir, topk_accuracy
    import numpy as np
    from pathlib import Path

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path="build-android",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{Path(pte_path).stem}",
        device_id="5f396958",
        soc_model="SM8750",
    )
    files = ["build-xnnpack/executor_runner"] if xnn else None
    custom_commands = (
        f"cd {adb.workspace} && ./executor_runner --model_path  {os.path.basename(adb.pte_path[0])} --input_list_path input_list.txt"
        if xnn else None
    )
    adb.push(inputs=inputs, input_list=input_list, files=files)
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


def mlm_eval(module, pte_path, inputs, input_list, targets, artifact_dir, xnn=False):
    from executorch.examples.qualcomm.utils import SimpleADB, make_output_dir
    import numpy as np
    from pathlib import Path
    import evaluate

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path="build-android",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{Path(pte_path).stem}",
        device_id="5f396958",
        soc_model="SM8750",
    )
    files = ["build-xnnpack/executor_runner"] if xnn else None
    custom_commands = (
        f"cd {adb.workspace} && ./executor_runner --model_path  {os.path.basename(adb.pte_path[0])} --input_list_path input_list.txt"
        if xnn else None
    )
    adb.push(inputs=inputs, input_list=input_list, files=files)
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


def get_model(model_name):
    from transformers import AutoModelForMaskedLM, AutoModelForImageClassification
    from executorch.examples.qualcomm.utils import get_imagenet_dataset, get_masked_language_model_dataset

    imagenet_dataset = "/local3/mnt/workspace/executorch_artifacts/dataset/imagenet_1k/imagenet-mini/val"
    path_to_wiki = "/local3/mnt/workspace/executorch_artifacts/dataset/wikisent2.txt"

    def get_MLM_sample_input(pretrained, data_size=10):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        return get_masked_language_model_dataset(path_to_wiki, tokenizer, data_size)

    def get_albert():
        from transformers import AlbertConfig
        pretrained = "albert/albert-base-v2"
        config = AlbertConfig.from_pretrained(pretrained)
        config.hidden_act = "gelu"
        inputs, targets, input_list = get_MLM_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained, config=config).eval()
        return module, inputs, targets, input_list, 16, 16, mlm_eval

    def get_bert():
        pretrained = "google-bert/bert-base-uncased"
        inputs, targets, input_list = get_MLM_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, input_list, 8, 16, mlm_eval

    def get_cvt():
        from transformers.models.cvt.modeling_cvt import CvtSelfAttention
        def attention_forward_without_einsum(self, hidden_state, height, width):
            if self.with_cls_token:
                cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
            batch_size, hidden_size, num_channels = hidden_state.shape
            # rearrange "b (h w) c -> b c h w"
            hidden_state = hidden_state.permute(0, 2, 1).view(
                batch_size, num_channels, height, width
            )

            key = self.convolution_projection_key(hidden_state)
            query = self.convolution_projection_query(hidden_state)
            value = self.convolution_projection_value(hidden_state)

            if self.with_cls_token:
                query = torch.cat((cls_token, query), dim=1)
                key = torch.cat((cls_token, key), dim=1)
                value = torch.cat((cls_token, value), dim=1)

            head_dim = self.embed_dim // self.num_heads

            query = self.rearrange_for_multi_head_attention(self.projection_query(query))
            key = self.rearrange_for_multi_head_attention(self.projection_key(key))
            value = self.rearrange_for_multi_head_attention(self.projection_value(value))
            # ====================Qualcomm Changed=================================
            attention_score = query @ key.transpose(-1, -2)
            attention_score = attention_score * self.scale
            # attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
            # =====================================================================
            attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
            attention_probs = self.dropout(attention_probs)
            # ====================Qualcomm Changed=================================
            context = attention_probs @ value
            # context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
            # =====================================================================
            # rearrange"b h t d -> b t (h d)"
            _, _, hidden_size, _ = context.shape
            context = (
                context.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size, hidden_size, self.num_heads * head_dim)
            )
            return context


        def _replace_attention(
            module: torch.nn.Module,
        ):
            for _, child in module.named_children():
                if isinstance(child, CvtSelfAttention):
                    child.forward = types.MethodType(  # pyre-ignore
                        attention_forward_without_einsum, child
                    )
                else:
                    _replace_attention(child)
            return module
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/cvt-13").eval()
        module = _replace_attention(module)
        return module, inputs, targets, input_list, 8, 8, image_eval

    def get_deit():
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224").eval()
        return module, inputs, targets, input_list, 8, 8, image_eval

    def get_efficientnet():
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0").eval()
        return module, inputs, targets, input_list, 16, 16, image_eval

    def get_eurobert():
        pretrained = "EuroBERT/EuroBERT-210m"
        inputs, targets, input_list = get_MLM_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained, trust_remote_code=True).eval()
        return module, inputs, targets, input_list, 16, 16, mlm_eval

    def get_distilbert():
        pretrained = "distilbert/distilbert-base-uncased"
        inputs, targets, input_list = get_MLM_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, input_list, 8, 16, mlm_eval

    def get_dit():
        def get_rvlcdip_dataset(data_size):
            from datasets import load_dataset
            from transformers import AutoImageProcessor
            from torch.utils.data import Dataset

            def get_data_loader():
                class DitDataset(Dataset):
                    def __init__(self, data_size) -> None:
                        self.data_size = data_size
                        self.dataset = self._get_dataset()
                        self.processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

                    def _get_dataset(self):
                        dataset = list(load_dataset("nielsr/rvl_cdip_10_examples_per_class", split="test"))
                        return dataset

                    def __getitem__(self, idx):
                        return (
                            self.processor(images=self.dataset[idx]["image"].convert("RGB"), return_tensors="pt"),
                            self.dataset[idx]["label"]
                        )

                    def __len__(self):
                        return len(self.dataset)

                dataset = DitDataset(data_size)
                torch.manual_seed(3407)
                return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)

            # prepare input data
            inputs, targets, input_list = [], [], ""
            for index, (feature, target) in enumerate(get_data_loader()):
                if index >= data_size:
                    break
                inputs.append((feature["pixel_values"],))
                targets.append(torch.tensor(target))
                input_list += f"input_{index}_0.raw\n"

            return inputs, targets, input_list

        inputs, targets, input_list = get_rvlcdip_dataset(100)
        module = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip").eval()
        return module, inputs, targets, input_list, 8, 8, image_eval

    def get_focalnet():
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/focalnet-tiny").eval()
        return module, inputs, targets, input_list, 8, 8, image_eval

    def get_mobilevit_v1():
        inputs, targets, input_list = mvit_get_imagenet_dataset(imagenet_dataset, 100)
        module = AutoModelForImageClassification.from_pretrained("apple/mobilevit-xx-small").eval()
        return module, inputs, targets, input_list, 16, 16, image_eval

    def get_mobilevit_v2():
        inputs, targets, input_list = mvit_get_imagenet_dataset(imagenet_dataset, 100)
        module = AutoModelForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").eval()
        return module, inputs, targets, input_list, 16, 16, image_eval

    def get_pvt():
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("Zetatech/pvt-tiny-224").eval()
        return module, inputs, targets, input_list, 8, 8, image_eval

    def get_roberta():
        pretrained = "xlm-roberta-base"
        inputs, targets, input_list = get_MLM_sample_input(pretrained)
        module = AutoModelForMaskedLM.from_pretrained(pretrained).eval()
        return module, inputs, targets, input_list, 8, 16, mlm_eval

    def get_swin():
        from transformers.models.swin import modeling_swin
        # Copy from transformers/models/swin/modeling_swin.py in transformers 4.47.1
        # (QCOM) Transform 6D dim to 5D dim
        def window_partition(input_feature, window_size):
            """
            Partitions the given input into windows.
            """
            batch_size, height, width, num_channels = input_feature.shape
            # ====================Qualcomm Changed=================================
            input_feature = input_feature.view(
                batch_size,
                height // window_size,
                window_size,
                width // window_size,
                window_size * num_channels,  # Merge the last two dimensions
            )
            windows = input_feature.permute(0, 1, 3, 2, 4).contiguous()
            windows = windows.view(-1, window_size, window_size, num_channels)
            # =====================================================================
            return windows

        # Copy from transformers/models/swin/modeling_swin.py in transformers 4.47.1
        # (QCOM) Transform 6D dim to 5D dim tests on huggingface version (4.47.1)
        def window_reverse(windows, window_size, height, width):
            """
            Merges windows to produce higher resolution features.
            """
            num_channels = windows.shape[-1]
            # ====================Qualcomm Changed=================================
            windows = windows.view(
                -1,
                height // window_size,
                width // window_size,
                window_size,
                window_size * num_channels,  # Merge the last two dimensions
            )
            windows = windows.permute(0, 1, 3, 2, 4).contiguous()
            windows = windows.view(-1, height, width, num_channels)
            # =====================================================================
            return windows

        # (QCOM) Replace the original window_partition and window_reverse functions
        # in the modeling_swin module with the new ones, due to QNN SDK does not support 6D tensor.
        modeling_swin.window_partition = window_partition
        modeling_swin.window_reverse = window_reverse
        inputs, targets, input_list = get_imagenet_dataset(imagenet_dataset, 100, (224, 224))
        module = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval()
        return module, inputs, targets, input_list, 8, 8, image_eval


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
        "mobilevit_v1": get_mobilevit_v1,
        "mobilevit_v2": get_mobilevit_v2,
        "pvt": get_pvt,
        "roberta": get_roberta,
        "swin": get_swin,
    }
    return model_dict[model_name]()


def generate_inputs(dest_path: str, list_file_name: str, inputs):
    input_list_file = ""

    # Prepare input data
    for idx, data in enumerate(inputs):
        cur_input = ""
        for i, d in enumerate(data):
            file_name = f"{dest_path}/input_{idx}_{i}.raw"
            cur_input += file_name + " "
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d)
            d.detach().numpy().tofile(file_name)
        input_list_file += cur_input.strip() + "\n"

    with open(f"{dest_path}/{list_file_name}", "w") as f:
        f.write(input_list_file)


if __name__ == "__main__":
    import argparse, subprocess, shutil, tempfile
    from executorch.backends.qualcomm.utils.utils import (
        from_context_binary,
        get_soc_to_chipset_map,
        QcomChipset,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--pte', '-p')
    parser.add_argument('--xnn', '-x', action="store_true")
    args = parser.parse_args()

    ws = "/local3/mnt/workspace"
    artifacts = f"{ws}/executorch_artifacts/ptes/qnn_native"
    pte_name = f"{artifacts}/{args.model}.pte"
    qnn_sdk = f"{ws}/qnn_sdk/2.36.0/qaisw-v2.36.0.250618071952_111824"
    #qnn_sdk = os.getenv("QNN_SDK_ROOT")

    with tempfile.TemporaryDirectory() as tmp_dir:
        module, inputs, targets, input_list, weight_bw, act_bw, eval_func = get_model(args.model)
        if args.pte is not None:
            eval_func(module, args.pte, inputs, input_list, targets, tmp_dir, args.xnn)
            exit(0)

        onnx_file = f"{tmp_dir}/{args.model}.onnx"
        onnx_program = torch.onnx.export(
            module, inputs[0], f=onnx_file, opset_version=18, dynamo=args.model in ['albert', 'bert', 'mobilevit_v2']
        )
        # pip install olive-ai
        from olive.model import ONNXModelHandler
        from olive.passes.olive_pass import create_pass_from_dict
        from olive.passes.onnx.graph_surgeries import GraphSurgeries

        ########## run olive passes to optimize onnx model #####################
        input_model = ONNXModelHandler(model_path=str(onnx_file))

        output_folder = tmp_dir
        p = create_pass_from_dict(
            GraphSurgeries,
            {"surgeries": [{"surgeon": "ReplaceAttentionMaskValue"}]},
            disable_search=True,
        )
        output_model = p.run(input_model, output_folder)
        onnx_file = output_model.model_path
        ########################################################################

        converted_dlc_file = f"{tmp_dir}/{args.model}.dlc"
        quantized_dlc_file = f"{tmp_dir}/{args.model}_quantized.dlc"
        # onnx_program.save(onnx_file)
        assert os.path.isfile(onnx_file)
        shutil.copy(onnx_file, artifacts)
        generate_inputs(tmp_dir, "input_list.txt", inputs)

        target = "x86_64-linux-clang"
        cmds = [
            f"source {ws}/miniconda3/bin/activate",
            "&& conda activate qnn",
            f"&& source {qnn_sdk}/bin/envsetup.sh",
            "&& qairt-converter",
            f"--input_network {onnx_file}",
            f"--output_path {converted_dlc_file}",
            "&& conda deactivate",
        ]
        print(f"execute {' '.join(cmds)}")
        subprocess.run(' '.join(cmds), shell=True, executable="/bin/bash",)
        assert os.path.isfile(converted_dlc_file)
        shutil.copy(converted_dlc_file, artifacts)

        cmds = [
            f"source {ws}/miniconda3/bin/activate",
            "&& conda activate qnn",
            f"&& source {qnn_sdk}/bin/envsetup.sh",
            f"&& source {qnn_sdk}/bin/envsetup.sh",
            f"&& {qnn_sdk}/bin/{target}/qairt-quantizer",
            f"--input_dlc {converted_dlc_file}",
            f"--output_dlc {quantized_dlc_file}",
            f"--input_list {tmp_dir}/input_list.txt",
            f"--weights_bitwidth {weight_bw}",
            f"--act_bitwidth {act_bw}",
            #"--bias_bitwidth 32",
            "--preserve_io_datatype",
			"--use_per_channel_quantization",
            "--use_native_input_files",
            #"--use_native_output_files",
            "&& conda deactivate",
        ]
        print(f"execute {' '.join(cmds)}")
        subprocess.run(' '.join(cmds), shell=True, executable="/bin/bash")
        assert os.path.isfile(quantized_dlc_file)
        shutil.copy(quantized_dlc_file, artifacts)

        config = {
            "backend_extension_config": {
                "backend_extensions": {
                    "shared_library_path": f"{qnn_sdk}/lib/{target}/libQnnHtpNetRunExtensions.so",
                    "config_file_path": f"{tmp_dir}/config.json",
                },
            },
            "config": {
                "graphs": [
                    {
                        "O": 3,
                        "graph_names":[f"{args.model}"],
                    }
                ],
                "devices": [
                    {
                        "cores": [
                            {"perf_profile": "burst", "rpc_control_latency": 100}
                        ],
                        "soc_id": int(get_soc_to_chipset_map()["SM8750"]),
                    }
                ]
            },
        }
        for file_name, data in config.items():
            with open(f"{tmp_dir}/{file_name}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
        cmds = [
            f"source {ws}/miniconda3/bin/activate",
            "&& conda activate qnn",
            f"&& source {qnn_sdk}/bin/envsetup.sh",
            f"&& source {qnn_sdk}/bin/envsetup.sh",
            f"&& {qnn_sdk}/bin/{target}/qnn-context-binary-generator",
            f"--backend {qnn_sdk}/lib/{target}/libQnnHtp.so",
            f"--model {qnn_sdk}/lib/{target}/libQnnModelDlc.so",
            f"--dlc_path {quantized_dlc_file}",
            f"--config_file {tmp_dir}/backend_extension_config.json",
            f"--binary_file {args.model}",
            f"--output_dir {tmp_dir}",
            "&& conda deactivate",
        ]
        print(f"execute {' '.join(cmds)}")
        subprocess.run(' '.join(cmds), shell=True, executable="/bin/bash")
        ctx_path = f"{tmp_dir}/{args.model}.bin"
        assert os.path.isfile(ctx_path)

        bundle_program = from_context_binary(
            ctx_path, "ctx_loader", soc_model=QcomChipset.SM8750
        )
        exec_prog = bundle_program["edge_program_manager"].to_executorch()
        with open(pte_name, "wb") as file:
            exec_prog.write_to_file(file)
        assert os.path.isfile(pte_name)

        #eval_func(module, pte_name, inputs, input_list, targets, tmp_dir)
