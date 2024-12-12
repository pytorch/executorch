import coremltools as ct
import argparse


if __name__ == "__main__":
    """
    Combines two CoreML models together
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
        "--output_path",
        type=str,
        help="Output path to save combined model",
    )

    args = parser.parse_args()
    model1_path = str(args.model1_path)
    model2_path = str(args.model2_path)
    output_path = str(args.output_path)


    desc = ct.utils.MultiFunctionDescriptor()

    desc.add_function(
        model1_path,
        src_function_name="main",
        target_function_name="model1"
    )
    desc.add_function(
        model2_path,
        src_function_name="main",
        target_function_name="model2"
    )
    desc.default_function_name = "model1"
    ct.utils.save_multifunction(desc, output_path)
