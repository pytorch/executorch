import argparse

import numpy as np
import torch
from torchao.quantization.utils import compute_error

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--golden",
    nargs="+",
    type=str,
)
parser.add_argument(
    "-o",
    "--output",
    nargs="+",
    type=str,
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
)
parser.add_argument(
    "-e",
    "--encoding",
    type=str,
)
args = parser.parse_args()
golden = [np.fromfile(f, dtype=np.float32) for f in args.golden]
output = [np.fromfile(f, dtype=eval(f"np.{args.dtype}")) for f in args.output]

with open(args.encoding, "r") as f:
    for i, g in enumerate(golden):
        enc = [float(x) for x in f.readline().split()]
        o = torch.from_numpy(output[i]).to(torch.float).sub(enc[1]).mul(enc[0])
        print(f"SQNR_{i}: {compute_error(torch.from_numpy(g), o)}")
