# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class SplitLinearModule(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        out_target_split_size=1,
        out_max_splits=1,
        in_target_split_size=1,
        in_max_splits=1,
    ):
        super(SplitLinearModule, self).__init__()
        self.out_split_sizes = self._get_split_sizes(
            out_features, out_target_split_size, out_max_splits
        )
        self.in_split_sizes = self._get_split_sizes(
            in_features, in_target_split_size, in_max_splits
        )
        print(
            f"Splitting out_features={out_features} into {len(self.out_split_sizes)} of size {self.out_split_sizes[0]}."
        )
        print(
            f"Splitting in_features={in_features} into {len(self.in_split_sizes)} of size {self.in_split_sizes[0]}."
        )

        # self.ops contains a list of linear ops for different pieces of the output matrix
        # The index of an op at (in_idx, out_idx) is given by self.op_index(in_idx, out_idx)
        self.ops = torch.nn.ModuleList()
        for idx_out, s_out in enumerate(self.out_split_sizes):
            for idx_in, s_in in enumerate(self.in_split_sizes):
                assert len(self.ops) == self.op_index(idx_in, idx_out)
                self.ops.append(torch.nn.Linear(s_in, s_out, bias=False))

    def op_index(self, in_index, out_index):
        idx = out_index * len(self.in_split_sizes) + in_index
        return idx

    def _get_split_sizes(self, n_features, target_split_size, max_splits):
        num_splits = max(n_features // target_split_size, 1)
        if num_splits > max_splits:
            num_splits = max_splits

        split_size = n_features // num_splits
        split_remainder = n_features % num_splits
        if split_remainder > 0:
            raise ValueError(
                f"Cannot split {n_features} with target_split_size={target_split_size} and max_splits={max_splits} because it leaves a remainder of {split_remainder}."
            )

        ret = [split_size for _ in range(num_splits)]
        return ret

    def set_params(self, weight):
        split_weights = []
        for w_out in weight.split(self.out_split_sizes, dim=0):
            for w in w_out.split(self.in_split_sizes, dim=1):
                split_weights.append(w)

        for i, split in enumerate(self.ops):
            split.weight = torch.nn.Parameter(split_weights[i])

    def forward(self, x):
        if len(self.in_split_sizes) == 1:
            out_chunks = [op(x) for op in self.ops]
        else:
            x_splits = x.split(self.in_split_sizes, dim=-1)
            out_chunks = [
                torch.sum(
                    torch.stack(
                        [
                            self.ops[self.op_index(in_idx, out_idx)].forward(
                                x_splits[in_idx]
                            )
                            for in_idx in range(len(self.in_split_sizes))
                        ],
                    ),
                    dim=0,
                )
                for out_idx in range(len(self.out_split_sizes))
            ]

        return torch.concat(out_chunks, dim=-1)


def replace_linear_with_split_linear(
    model, out_target_split_size, out_max_splits, in_target_split_size, in_max_splits=1, fqn_filer=None,
):
    if fqn_filer is None:
        fqn_filer = lambda fqn: True

    for name, module in model.named_children():
        should_split = isinstance(module, torch.nn.Linear) and fqn_filer(name)
        print("TESTING", name, "WILL SPLIT", should_split)
        if should_split:
            assert module.bias is None, "SplitLinearModule does not support bias"
            new_module = SplitLinearModule(
                module.in_features,
                module.out_features,
                out_target_split_size,
                out_max_splits,
                in_target_split_size,
                in_max_splits,
            )
            new_module.set_params(module.weight)
            setattr(model, name, new_module)
        else:
            replace_linear_with_split_linear(
                module,
                out_target_split_size,
                out_max_splits,
                in_target_split_size,
                in_max_splits,
                fqn_filer,
            )
