from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
import torch
import os
import numpy as np

M, K, N = 2048, 8192, 1
bits = 4
group_size = 128

qlinear = TorchQuantLinear(
    bits=bits,
    group_size=group_size,
    sym=True,
    desc_act=False,
    in_features=K,
    out_features=M,
    bias=False,
)
data_path = f"m{M}_k{K}_g{group_size}/"
qlinear.load_state_dict(torch.load(os.path.join(data_path, "qlinear.pt")))
qlinear.post_init()

x = torch.from_numpy(np.fromfile(os.path.join(data_path, "x.bin"), dtype=np.float16)).reshape(N, K)
y_ref = qlinear.forward(x)

from executorch.backends.qualcomm.builders.utils import unpack_gptqv2, hvx_preprocess_weights
w, scales, zeros, _, _, _ = unpack_gptqv2(qlinear.qweight.numpy(), qlinear.scales.numpy(), qlinear.qzeros.numpy())
w.tofile(os.path.join(data_path, "w_unpacked.bin"))
scales.tofile(os.path.join(data_path, "s_unpacked.bin"))
zeros.tofile(os.path.join(data_path, "z_unpacked.bin"))

w_dq = w.T.reshape(K // group_size, group_size, M).astype(np.float16) - (2 ** (bits - 1))
w_dq = w_dq.transpose(1, 0, 2) * scales.T
w_dq = w_dq - zeros.T
w_dq = w_dq.transpose(1, 0, 2).reshape(K, M)

y_ref2 = x.numpy().dot(w_dq)

qweight_repacked, scales_repacked = hvx_preprocess_weights(w, scales, zeros, bits, tile_p=M*bits)
qweight_repacked.tofile(os.path.join(data_path, "w_repacked.bin"))
scales_repacked.tofile(os.path.join(data_path, "s_repacked.bin"))
