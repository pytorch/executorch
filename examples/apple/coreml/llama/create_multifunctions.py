#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import coremltools as ct


def extract_models(pte_path: str, output_dir: str) -> list[str]:
    """
    Extract CoreML models from a PTE file.
    Returns list of paths to extracted .mlpackage files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the extraction script
    script_path = Path(__file__).parent.parent / "scripts" / "extract_coreml_models.py"
    
    # Save current directory and change to output dir (extract script outputs to cwd)
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "-m", pte_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error extracting models: {result.stderr}")
            sys.exit(1)
        print(result.stdout)
    finally:
        os.chdir(original_cwd)
    
    # Find extracted mlpackage files
    extracted_dir = Path(output_dir) / "extracted_coreml_models"
    
    # Debug: print what we find
    print(f"  Looking in: {extracted_dir}")
    for model_dir in sorted(extracted_dir.iterdir()):
        print(f"    {model_dir.name}/")
        if model_dir.is_dir():
            for item in list(model_dir.iterdir())[:10]:
                print(f"      {item.name}")
    
    model_paths = []
    for model_dir in sorted(extracted_dir.iterdir()):
        if model_dir.is_dir():
            # Look for .mlpackage inside the model directory
            found = False
            for item in model_dir.iterdir():
                if item.suffix == ".mlpackage":
                    model_paths.append(str(item))
                    found = True
                    break
            
            # If no .mlpackage found, check for lowered_module directory
            if not found:
                lowered_module = model_dir / "lowered_module"
                if lowered_module.exists() and lowered_module.is_dir():
                    # Debug: show contents of lowered_module
                    print(f"    Contents of {lowered_module}:")
                    for item in list(lowered_module.iterdir())[:10]:
                        print(f"      {item.name}")
                    
                    # Look for .mlpackage inside lowered_module
                    for item in lowered_module.iterdir():
                        if item.suffix == ".mlpackage":
                            model_paths.append(str(item))
                            found = True
                            break
                    
                    # If still not found, look for model.mlmodel file
                    if not found:
                        mlmodel_file = lowered_module / "model.mlmodel"
                        if mlmodel_file.exists():
                            # Load and save as mlpackage
                            mlpackage_path = model_dir / f"{model_dir.name}.mlpackage"
                            model = ct.models.MLModel(str(mlmodel_file))
                            model.save(str(mlpackage_path))
                            model_paths.append(str(mlpackage_path))
                            found = True
    
    return model_paths


def create_multifunction_model(
    prefill_mlpackage: str,
    decode_mlpackage: str,
    output_path: str,
    compile_model: bool
) -> str:
    """
    Create a multifunction model combining prefill and decode.
    Returns the path to the output model.
    """
    desc = ct.utils.MultiFunctionDescriptor()
    
    desc.add_function(
        prefill_mlpackage,
        src_function_name="main",
        target_function_name="prefill"
    )
    desc.add_function(
        decode_mlpackage,
        src_function_name="main",
        target_function_name="decode"
    )
    
    desc.default_function_name = "decode"
    
    if compile_model:
        # Save mlpackage first, then compile
        mlpackage_path = output_path + ".mlpackage"
        ct.utils.save_multifunction(desc, mlpackage_path)
        
        compiled_path = ct.utils.compile_model(mlpackage_path)
        dest_path = output_path + ".mlmodelc"
        
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(compiled_path, dest_path)
        
        # Clean up intermediate mlpackage
        shutil.rmtree(mlpackage_path)
        
        print(f"Saved compiled model to {dest_path}")
        return dest_path
    else:
        mlpackage_path = output_path + ".mlpackage"
        ct.utils.save_multifunction(desc, mlpackage_path)
        print(f"Saved model to {mlpackage_path}")
        return mlpackage_path


def main():
    parser = argparse.ArgumentParser(
        description="Create multifunction CoreML models from prefill/decode PTE files"
    )
    parser.add_argument(
        "--prefill_model",
        required=True,
        help="Path to the prefill PTE file"
    )
    parser.add_argument(
        "--decode_model",
        required=True,
        help="Path to the decode PTE file"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Compile the models to .mlmodelc format"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for the multifunction models (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Create temp directories for extraction
    temp_dir = Path(args.output_dir) / "temp_extraction"
    prefill_extract_dir = temp_dir / "prefill"
    decode_extract_dir = temp_dir / "decode"
    
    print("Extracting prefill models...")
    prefill_models = extract_models(args.prefill_model, str(prefill_extract_dir))
    print(f"Found {len(prefill_models)} prefill models")
    
    print("Extracting decode models...")
    decode_models = extract_models(args.decode_model, str(decode_extract_dir))
    print(f"Found {len(decode_models)} decode models")
    
    if len(prefill_models) != len(decode_models):
        print(f"Error: Number of prefill models ({len(prefill_models)}) does not match decode models ({len(decode_models)})")
        sys.exit(1)
    
    num_models = len(prefill_models)
    print(f"\nCreating {num_models} multifunction models...")
    
    # Create multifunction models (mod1, mod2, mod3, ...)
    for i in range(num_models):
        model_num = i + 1
        output_path = str(Path(args.output_dir) / f"mod{model_num}")
        
        print(f"\nCreating mod{model_num}...")
        print(f"  Prefill: {prefill_models[i]}")
        print(f"  Decode: {decode_models[i]}")
        
        create_multifunction_model(
            prefill_mlpackage=prefill_models[i],
            decode_mlpackage=decode_models[i],
            output_path=output_path,
            compile_model=args.compile
        )
    
    # Clean up temp directory
    print("\nCleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        print(f"Warning: Could not fully clean up temp directory: {e}")
        print(f"You may want to manually delete: {temp_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
