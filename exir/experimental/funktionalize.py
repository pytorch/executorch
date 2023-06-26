# A horrible implementation of functionalization, DO NOT use this!

import torch
from executorch.exir.graph_module import (
    EXIR_METADATA,
    get_exir_meta,
    make_export_graph_module,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.tracer import unwrap_functional
from functorch.experimental import functionalize
from torch import fx
from torch._functorch.eager_transforms import _assert_wrapped_functional
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _get_current_dispatch_mode


def make_funktionalized(gm, flatten=False):
    def f(*args):
        with torch.fx.traceback.preserve_node_meta():
            ret = RecursiveInterpreter(gm).run(*args)
            if flatten:
                ret, _ = pytree.tree_flatten(ret)
            return ret

    return functionalize(f, remove="mutations_and_views")


class SkippingInterpreter(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.cond:
            pred, true, false, params = args
            return true(*params)
        elif target == torch.ops.map:
            f, xs, *params = args
            sample_out = f(xs[0], *params)
            return sample_out.new_empty([xs.shape[0], *sample_out.shape])
        else:
            return super().call_function(target, args, kwargs)


class RecursiveInterpreter(torch.fx.Interpreter):
    def run_node(self, node, *args, **kwargs):
        self.node = node
        return super().run_node(node, *args, **kwargs)

    def call_function(self, target, args, kwargs):
        if len(self.node.users) == 0 and target in (
            torch.ops.aten.empty.memory_format,
        ):
            return
        mode = _get_current_dispatch_mode()

        def _unwrap_proxy(e):
            if isinstance(e, torch.nn.Module):
                if not isinstance(e, torch.fx.GraphModule):
                    return None
                next_name = None
                i = 0
                while not next_name:
                    candidate = f"argmodule_{i}"
                    if hasattr(mode.tracer.root, candidate):
                        i += 1
                    else:
                        next_name = candidate
                mode.tracer.root.register_module(next_name, e)
            return unwrap_proxy(mode, e)

        def unwrap_real(e):
            with disable_proxy_modes_tracing():
                return e.clone()

        if target == torch.ops.cond:
            pred, true, false, params = args
            params, _ = pytree.tree_flatten(params)
            unwrapped_params = pytree.tree_map_only(
                torch.Tensor, unwrap_functional, params
            )
            with disable_proxy_modes_tracing():
                true_gm = make_fx(make_funktionalized(true))(
                    *pytree.tree_map_only(torch.Tensor, unwrap_real, unwrapped_params),
                )
                false_gm = make_fx(make_funktionalized(false))(
                    *pytree.tree_map_only(torch.Tensor, unwrap_real, unwrapped_params),
                )
            proxy_args = pytree.tree_map(
                _unwrap_proxy,
                (
                    unwrap_functional(pred) if isinstance(pred, torch.Tensor) else pred,
                    true_gm,
                    false_gm,
                    unwrapped_params,
                ),
            )
            proxy_out = mode.tracer.create_proxy(
                "call_function", target, proxy_args, {}, name="conditional"
            )
            with disable_proxy_modes_tracing():
                out = SkippingInterpreter(true_gm).run(*params)
                unwrapped = unwrap_functional(out)
                assert isinstance(unwrapped, FakeTensor)
                assert hasattr(unwrapped, "fake_mode")
                track_tensor_tree(
                    unwrapped, proxy_out, constant=None, tracer=mode.tracer
                )
                _assert_wrapped_functional(unwrapped, out)
            return out
        elif target == torch.ops.map:
            f, xs, *params = args
            unwrapped_params = pytree.tree_map_only(
                torch.Tensor, unwrap_functional, params
            )
            with disable_proxy_modes_tracing():
                sample = xs[0]
                body_gm = make_fx(make_funktionalized(f))(
                    *pytree.tree_map_only(
                        torch.Tensor,
                        unwrap_real,
                        (unwrap_functional(sample), *unwrapped_params),
                    ),
                )

            proxy_args = pytree.tree_map(
                _unwrap_proxy,
                (body_gm, unwrap_functional(xs), *unwrapped_params),
            )
            proxy_out = mode.tracer.create_proxy(
                "call_function", target, proxy_args, {}, name="map"
            )
            with disable_proxy_modes_tracing():
                sample_out = SkippingInterpreter(body_gm).run(xs[0], *params)
                out = sample_out.new_empty([xs.shape[0], *sample_out.shape])
                unwrapped = unwrap_functional(out)
                assert isinstance(unwrapped, FakeTensor)
                assert hasattr(unwrapped, "fake_mode")
                track_tensor_tree(
                    unwrapped, proxy_out, constant=None, tracer=mode.tracer
                )
                _assert_wrapped_functional(unwrapped, out)
            return out
        elif target in (torch.ops.aten._native_batch_norm_legit.default,):
            assert args[5] is False
            res = super().call_function(
                torch.ops.aten._native_batch_norm_legit_functional.default,
                args,
                kwargs,
            )
            return res
        else:
            return super().call_function(target, args, kwargs)


def funktionalize(graph, inputs):
    gm = make_fx(
        make_funktionalized(graph, flatten=True),
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )(*pytree.tree_flatten(inputs)[0])
    replacements = {
        torch.ops.aten.view.default: torch.ops.aten.view_copy.default,
        torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
        torch.ops.aten.transpose.int: torch.ops.aten.transpose_copy.int,
        torch.ops.aten.expand.default: torch.ops.aten.expand_copy.default,
        torch.ops.aten.unsqueeze.default: torch.ops.aten.unsqueeze_copy.default,
    }
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op == "call_function":
                if node.target in replacements:
                    assert len(node.users) == 1
                    node.target = replacements[node.target]

        module.recompile()

    ret = make_export_graph_module(gm, gm.graph)
    ret.meta[EXIR_METADATA] = get_exir_meta(graph)
    return ret


class FunktionalizationPass(ExportPass):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        return PassResult(funktionalize(graph_module, self.args), True)
