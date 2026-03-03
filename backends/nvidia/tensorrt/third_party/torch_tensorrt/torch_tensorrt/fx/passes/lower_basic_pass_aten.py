import logging
import operator
from typing import Any

import torch
import torch.fx
from torch.fx.experimental.const_fold import split_const_subgraphs
from torch.fx.passes.infra.pass_base import PassResult

_LOGGER = logging.getLogger(__name__)

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module.
    def skip_folding(node: torch.fx.Node):
        if node.target == torch.ops.aten.sym_size:
            return True

    const_split_mod = split_const_subgraphs(
        traced_mod, skip_folding_node_fn=skip_folding
    )
    const_split_mod.run_folding()
    return const_split_mod


def replace_inplace_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    Remove this func after functionalization is workable
    """
    modified = False
    map_func = {
        torch.ops.aten.relu_.default: torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh_.default: torch.ops.aten.hardtanh.default,
        torch.ops.aten.add_.Tensor: torch.ops.aten.add.Tensor,
    }
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in map_func.keys():
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_args = node.args
                new_node = module.graph.create_node(
                    "call_function",
                    map_func[node.target],
                    args=new_args,
                    kwargs=None,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_native_layernorm_with_layernorm(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if (
            n.op == "call_function"
            and n.target == torch.ops.aten.native_layer_norm.default
        ):
            for v in n.users:
                if v.op == "call_function" and v.target == operator.getitem:
                    if v.args[1] != 0:
                        raise RuntimeError(
                            f"Got args[{v.args[1]}]!!\n"
                            "layernorm can only generate output (args[0]), "
                            "not mean (args[1]) or std (args[2])!"
                        )
                    new_op = torch.ops.aten.layer_norm.default
                    new_args = (*n.args, True)  # cudnn_enable=True
                    modified = True
                else:
                    continue

                with module.graph.inserting_after(v):
                    new_node = module.graph.create_node(
                        "call_function",
                        new_op,
                        args=new_args,
                        kwargs=v.kwargs,
                    )
                    v.replace_all_uses_with(new_node)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_transpose_mm_op_with_linear(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.t.default:
            to_erase = []
            for v in n.users:
                if v.op == "call_function" and v.target == torch.ops.aten.addmm.default:
                    new_op = torch.ops.aten.linear
                    bias, inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, bias)
                    modified = True
                elif v.op == "call_function" and v.target == torch.ops.aten.mm.default:
                    new_op = torch.ops.aten.linear
                    inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, None)
                    modified = True
                # this pass should be after `compose_bmm`
                elif v.op == "call_function" and v.target == aten_compose_bmm_2d:
                    new_op = torch.ops.aten.linear
                    inp, _ = list(v.args)
                    weight = list(n.args)[0]
                    new_args = (inp, weight, None)
                    modified = True
                else:
                    continue

                with module.graph.inserting_after(v):
                    new_node = module.graph.create_node(
                        "call_function",
                        new_op,
                        args=new_args,
                        kwargs=v.kwargs,
                    )
                    v.replace_all_uses_with(new_node)
                    to_erase.append(v)
            for v in to_erase:
                module.graph.erase_node(v)
    module.graph.eliminate_dead_code()
    module.recompile()
    # handle the linear with multiple dim, remove the extra reshape
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.linear:
            before = n.args[0]
            after = next(iter(n.users))
            if (len(n.users) == 1 and after.target == torch.ops.aten.view.default) and (
                before.target == torch.ops.aten.view.default and len(before.users) == 1
            ):
                real_input = before.args[0]
                new_args = list(n.args)
                new_args[0] = real_input
                n.args = tuple(new_args)
                after.replace_all_uses_with(n)
                module.graph.eliminate_dead_code()
                module.recompile()

    return PassResult(module, modified)


def replace_aten_op_with_indices(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.native_batch_norm.default,
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten._native_batch_norm_legit_no_training.default,
        ):
            modified = True
            if len(n.users) != 1:
                raise RuntimeError(
                    f"{n.target} has users={len(n.users)}. We can only handle it with 1 user"
                )
            if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                new_op = torch.ops.aten.max_pool2d
                new_args = n.args
            elif n.target == torch.ops.aten.max_pool3d_with_indices.default:
                new_op = torch.ops.aten.max_pool3d
                new_args = n.args
            elif (
                n.target == torch.ops.aten.native_batch_norm.default
                or n.target == torch.ops.aten._native_batch_norm_legit.default
            ):
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                new_args = tuple(new_args)
            elif (
                n.target == torch.ops.aten._native_batch_norm_legit_no_training.default
            ):
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                # _native_batch_norm_legit_no_training doesn't take in a training arg (assumed to be false)
                # but batchnorm takes in a training arg at position 5.
                new_args.insert(5, False)
                new_args = tuple(new_args)

            getitem_node = next(iter(n.users))
            with module.graph.inserting_after(getitem_node):
                new_node = module.graph.create_node(
                    "call_function",
                    new_op,
                    args=new_args,
                    kwargs=n.kwargs,
                )
                getitem_node.replace_all_uses_with(new_node)
                module.graph.erase_node(getitem_node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def replace_aten_reshape_alias_with_replace(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    # The stride parameter is not used. Replace with reshape without stride
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten._reshape_alias.default,
        ):
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_args = (node.args[0], node.args[1])
                new_node = module.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=new_args,
                    kwargs=None,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
            break
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def remove_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    1. Remove clone, _unsafe_view node. #TODO Remove this func after functionalization is workable
    2. Remove inefficient op getitem(index=slice) P561572458
    """
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (torch.ops.aten.clone.default,):
            modified = True
            node = n
            input_n = node.all_input_nodes[0]
            node.replace_all_uses_with(input_n)
    module.graph.eliminate_dead_code()
    module.recompile()
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten._unsafe_view.default,
        ):
            modified = True
            node = n
            with module.graph.inserting_after(node):
                new_node = module.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=node.args,
                    kwargs=node.kwargs,
                )
                node.replace_all_uses_with(new_node)
                module.graph.erase_node(node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def aten_operator_getitem(*args):
    return operator.getitem(*args)


def replace_builtin_ops(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    To differential the same op in fx2ait as they are registered in the same dictionary
    """

    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (operator.getitem,):
            modified = True
            n.target = aten_operator_getitem
    module.graph.eliminate_dead_code()
    module.recompile()

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


###############
"""
Trace compose. For some ops, we do not want to decompose further but want coarse granularity
For ex:
1. bmm
2. chunk
3. getitem(input, idx=(slice(),slice()...))
"""


def aten_compose_getitem_slice(input, list_args):
    for _, args in enumerate(list_args):
        input = torch.ops.aten.slice.Tensor(input, *args)
    return input


def compose_getitem_slice(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed getitem(input, idx=(slice(),slice()...))
    """

    def match_pattern(module, node):
        if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor:
            holder = []
            holder.append(node)
            while (
                len(node.users.keys()) == 1
                and next(iter(node.users)).target == torch.ops.aten.slice.Tensor
                and node.args[1] + 1 == next(iter(node.users)).args[1]
            ):
                node = next(iter(node.users))
                holder.append(node)
            if len(holder) == 1:
                return (False,)
            else:
                return (True, holder)
        return (False,)

    modified = False
    for node in module.graph.nodes:
        res = match_pattern(module, node)
        if res[0]:
            modified = True
            holder = res[1]
            input_n = holder[0].args[0]
            last_n = holder[-1]
            list_args = []
            for h_n in holder:
                list_args.append(h_n.args[1:])

            with module.graph.inserting_after(last_n):
                new_args = (input_n, list_args)
                new_node = module.graph.create_node(
                    "call_function",
                    aten_compose_getitem_slice,
                    args=new_args,
                    kwargs=None,
                )
            last_n.replace_all_uses_with(new_node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def aten_compose_bmm_2d(flat_args_1, flat_args_2):
    sym_size = torch.ops.aten.sym_size(flat_args_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(flat_args_1, 1)
    sym_size_2 = torch.ops.aten.sym_size(flat_args_1, 2)
    expand = torch.ops.aten.expand.default(
        flat_args_1, [sym_size, sym_size_1, sym_size_2]
    )
    view = torch.ops.aten.view.default(expand, [sym_size, sym_size_1, sym_size_2])
    sym_size_3 = torch.ops.aten.sym_size(flat_args_2, 0)
    sym_size_4 = torch.ops.aten.sym_size(flat_args_2, 1)
    expand_1 = torch.ops.aten.expand.default(
        flat_args_2, [sym_size, sym_size_3, sym_size_4]
    )
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size, sym_size_3, sym_size_4])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.view.default(bmm, [sym_size, sym_size_1, sym_size_4])
    return view_2


def aten_compose_bmm_3d(flat_args_1, flat_args_2):
    sym_size = torch.ops.aten.sym_size(flat_args_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(flat_args_1, 1)
    sym_size_2 = torch.ops.aten.sym_size(flat_args_1, 2)
    expand = torch.ops.aten.expand.default(
        flat_args_1, [sym_size, sym_size_1, sym_size_2]
    )
    view = torch.ops.aten.view.default(expand, [sym_size, sym_size_1, sym_size_2])
    sym_size_3 = torch.ops.aten.sym_size(flat_args_2, 1)
    sym_size_4 = torch.ops.aten.sym_size(flat_args_2, 2)
    expand_1 = torch.ops.aten.expand.default(
        flat_args_2, [sym_size, sym_size_3, sym_size_4]
    )
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size, sym_size_3, sym_size_4])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.view.default(bmm, [sym_size, sym_size_1, sym_size_4])
    return view_2


def compose_bmm(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed bmm (matmul)
    """
    modified = False
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (torch.ops.aten.bmm.default,):
            modified = True
            node = n
            input_n = node.all_input_nodes[0]
            other_n = node.all_input_nodes[1]
            output = next(iter(node.users))
            input_input_n = input_n.all_input_nodes[0]
            if (
                input_input_n.target != torch.ops.aten.expand.default
                and input_n.target != torch.ops.aten.view.default
            ):
                raise RuntimeError(
                    "Bmm is addressed in fixed pattern. A new pattern is met!"
                )
            real_input = input_input_n.all_input_nodes[0]
            input_other_n = other_n.all_input_nodes[0]
            if (
                input_other_n.target != torch.ops.aten.expand.default
                and other_n.target != torch.ops.aten.view.default
            ):
                raise RuntimeError(
                    "Bmm is addressed in fixed pattern. A new pattern is met!"
                )
            real_other = input_other_n.all_input_nodes[0]
            if len(real_other.meta["val"].size()) == 2:
                new_func = aten_compose_bmm_2d
            if len(real_other.meta["val"].size()) == 3:
                new_func = aten_compose_bmm_3d

            with module.graph.inserting_after(node):
                new_args = (real_input, real_other)
                new_node = module.graph.create_node(
                    "call_function",
                    new_func,
                    args=new_args,
                    kwargs=None,
                )
            output.replace_all_uses_with(new_node)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)


def aten_compose_chunk(flat_args_1, chunk, dim):
    sym_size = torch.ops.aten.sym_size(flat_args_1, dim)
    add = operator.add(sym_size, chunk)
    sub = operator.sub(add, 1)
    floordiv = operator.floordiv(sub, chunk)
    split = torch.ops.aten.split.Tensor(flat_args_1, floordiv, dim)
    return split


def compose_chunk(
    module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    combine decomposed chunk
    """

    def match_pattern(module, node):
        if node.op == "call_function" and node.target in (torch.ops.aten.split.Tensor,):
            div = node.args[1]
            input = node.args[0]
            if isinstance(div, int):
                return (False,)
            if div.target != operator.floordiv:
                return (False,)
            else:
                div_const = div.args[1]
                sub = div.args[0]
                if sub.target != operator.sub:
                    return (False,)
                else:
                    add = sub.args[0]
                    if add.target != operator.add:
                        return (False,)
                    else:
                        add_const = add.args[1]
                        if add_const != div_const:
                            return (False,)
                        symsize = add.args[0]
                        if symsize.target != torch.ops.aten.sym_size:
                            return (False,)
                        else:
                            symsize_input = symsize.args[0]
                            dim = symsize.args[1]
                            if symsize_input != input:
                                return (False,)

            return (True, div_const, dim)
        else:
            return (False,)

    modified = False
    for node in module.graph.nodes:
        res = match_pattern(module, node)
        if res[0]:
            modified = True
            with module.graph.inserting_after(node):
                new_args = (node.args[0], res[1], res[2])
                new_node = module.graph.create_node(
                    "call_function",
                    aten_compose_chunk,
                    args=new_args,
                    kwargs=None,
                )
            node.replace_all_uses_with(new_node)

    module.graph.eliminate_dead_code()
    module.recompile()
    return PassResult(module, modified)
