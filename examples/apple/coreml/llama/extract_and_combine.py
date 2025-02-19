import argparse
import os
import shutil
import subprocess

import coremltools as ct

if __name__ == "__main__":
    """
    Extract mlpackage from two CoreML pte files, and combine them into one mlpackage using multifunction
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m1",
        "--model1_path",
        type=str,
        help="Model1 path.",
    )
    parser.add_argument(
        "-m2",
        "--model2_path",
        type=str,
        help="Model2 path.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output path to save combined model",
    )

    args = parser.parse_args()
    model1_path = str(args.model1_path)
    model2_path = str(args.model2_path)
    output_dir = str(args.output_dir)

    if os.path.exists(output_dir):
        raise Exception(
            f"Output directory {output_dir} already exists.  Please make delete it before running script."
        )
    os.makedirs(output_dir)

    coreml_extract_path = os.path.join(os.getcwd(), "extracted_coreml_models")
    if os.path.exists(coreml_extract_path):
        raise Exception(
            f"{coreml_extract_path} already exists.  Please delete it before running script."
        )

    extract_script_path = os.path.join(
        os.path.dirname(__file__), "../scripts/extract_coreml_models.py"
    )
    extracted_path = "extracted_coreml_models/model_1/lowered_module/model.mlpackage"

    subprocess.run(["python", extract_script_path, "--model", model1_path])
    items = os.listdir("extracted_coreml_models")
    assert len(items) == 1, "Expected one CoreML partition"
    shutil.copytree(extracted_path, f"{output_dir}/model1.mlpackage")

    subprocess.run(["python", extract_script_path, "--model", model2_path])
    items = os.listdir("extracted_coreml_models")
    assert len(items) == 1, "Expected one CoreML partition"
    shutil.copytree(extracted_path, f"{output_dir}/model2.mlpackage")

    desc = ct.utils.MultiFunctionDescriptor()

    desc.add_function(
        f"{output_dir}/model1.mlpackage",
        src_function_name="main",
        target_function_name="model1",
    )
    desc.add_function(
        f"{output_dir}/model2.mlpackage",
        src_function_name="main",
        target_function_name="model2",
    )
    desc.default_function_name = "model1"
    ct.utils.save_multifunction(desc, f"{output_dir}/combined.mlpackage")
