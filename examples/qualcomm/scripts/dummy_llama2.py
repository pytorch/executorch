# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys
from multiprocessing.connection import Client

import numpy as np
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.llama2 import Llama2Model
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)


def create_device_inputs(example_inputs, use_kv_cache):
    inputs = [inp.to(torch.int32) for inp in example_inputs]
    input_list = ""
    if use_kv_cache:
        for i, d in enumerate(inputs[0]):
            if type(d) == list:
                d = torch.stack(d)
            d.numpy().tofile(f"{args.artifact}/input_0_0.raw")
            input_list = f"input_0_{i}.raw "
    else:
        inputs[0].numpy().tofile(f"{args.artifact}/input_0_0.raw")
        input_list = "input_0_0.raw"
    input_list += "\n"
    return tuple(inputs), input_list


if __name__ == "__main__":
    print(
        "[WARNING] The module of llama is changing frequently. This script might not work"
    )
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./dummy_llama2",
        default="./dummy_llama2",
        type=str,
    )

    # TODO kv cache is not yet enabled
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 8bits activation and 8bits weight. Support 8a8w, 16a16w and 16a4w.",
        default="8a8w",
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama2 checkpoint.",
        default=False,
    )

    parser.add_argument(
        "--params",
        help="Pass llama2 params json file.",
        default=False,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if args.params and args.checkpoint:
        instance = Llama2Model(
            use_kv_cache=args.use_kv_cache,
            checkpoint=args.checkpoint,
            params=args.params,
        )
    else:
        instance = Llama2Model(
            use_kv_cache=args.use_kv_cache,
        )

    inputs, input_list = create_device_inputs(
        instance.get_example_inputs(), args.use_kv_cache
    )

    pte_filename = "dummy_llama2_qnn"

    if args.ptq == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
    elif args.ptq == "16a16w":
        quant_dtype = QuantDtype.use_16a16w
    elif args.ptq == "16a4w":
        quant_dtype = QuantDtype.use_16a4w
    else:
        raise AssertionError(
            f"No support for quant type {args.ptq}. Support 8a8w, 16a16w and 16a4w."
        )

    if args.use_fp16:
        quant_dtype = None

    build_executorch_binary(
        instance.get_eager_model().eval(),
        inputs,
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        custom_annotations=(),
        quant_dtype=quant_dtype,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        sys.exit(0)

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        artifact_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    output_raws = []

    def post_process():
        for f in sorted(
            os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
        ):
            filename = os.path.join(output_data_folder, f)
            if re.match(r"^output_[0-9]+_[1-9].raw$", f):
                os.remove(filename)
            else:
                output = np.fromfile(filename, dtype=np.float32)
                output_raws.append(output)

    adb.pull(output_path=args.artifact, callback=post_process)

    x86_golden = instance.get_eager_model().eval()(inputs[0])
    device_output = torch.from_numpy(output_raws[0]).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                    }
                )
            )
    else:
        print(f"is_close? {result}")
        print(f"x86_golden {x86_golden}")
        print(f"device_out {device_output}")
