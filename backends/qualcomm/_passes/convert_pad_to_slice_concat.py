import operator
import torch
from torch.fx import GraphModule
from executorch.exir.pass_base import ExportPass, PassResult

import operator
import torch
from torch.fx import GraphModule
from executorch.exir.pass_base import ExportPass, PassResult

class ConvertPadToSliceConcat(ExportPass):
    """
    Replace aten.pad(..., mode in {'circular','replicate'}) with slice+cat (+expand for replicate).
    Supports 1D/2D (NCL / NCHW-like). SymInt-safe for torch.export graphs.
    """

    def __init__(self):
        super().__init__()

    # ---------- small helpers ----------

    def _copy_meta(self, src, dst, val_transform=None):
        dst.meta = dict(getattr(src, "meta", {}))
        if "val" in getattr(src, "meta", {}) and isinstance(src.meta["val"], torch.Tensor):
            v = src.meta["val"]
            if val_transform is not None:
                try:
                    v = val_transform(v)
                except Exception:
                    pass
            dst.meta["val"] = v

    def _set_scalar_meta(self, node, dtype=torch.int64):
        node.meta = getattr(node, "meta", {})
        node.meta["val"] = torch.tensor(0, dtype=dtype)

    def _sym_size(self, graph, x, dim):
        if hasattr(torch.ops.aten, "sym_size"):
            n = graph.create_node("call_function", torch.ops.aten.sym_size.int, (x, dim))
        else:
            n = graph.create_node("call_function", torch.ops.aten.size.int, (x, dim))
        self._set_scalar_meta(n)
        return n

    def _sym_sub(self, graph, a, b):
        n = graph.create_node("call_function", operator.sub, (a, b))
        self._set_scalar_meta(n)
        return n

    def _rank_from_meta(self, t):
        r = None
        if hasattr(t, "meta") and isinstance(t.meta.get("val", None), torch.Tensor):
            r = t.meta["val"].dim()
        return r

    def _expand_along_dim(self, graph, t, dim, new_len, before):
        """
        Build aten.expand(t, new_sizes) where only 'dim' changes to new_len.
        Works with SymInt sizes. new_len is a python int.
        """
        with graph.inserting_before(before):
            rank = self._rank_from_meta(t)
            if rank is None:
                # Fallback: grab sizes with sym_size one-by-one assuming up to 8 dims
                # (most models are 4D here; if meta is missing, 4 is reasonable)
                rank = 4
            sizes = []
            # convert negative dim to pos
            pdim = dim % rank
            for d in range(rank):
                if d == pdim:
                    sizes.append(int(new_len))
                else:
                    sizes.append(self._sym_size(graph, t, d))
            n = graph.create_node("call_function", torch.ops.aten.expand.default, (t, sizes))
            # meta: broadcast view to the new shape if we have it
            def _vt(v):
                shape = list(v.shape)
                shape[pdim] = int(new_len)
                return v.expand(shape)
            self._copy_meta(t, n, _vt)
            return n

    # ---------- main entry ----------

    def call(self, gm: GraphModule) -> PassResult:
        g = gm.graph
        modified = False

        for node in list(g.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.pad.default:
                # args: (x, pad, mode, [value])
                if len(node.args) < 3 or not isinstance(node.args[2], str):
                    continue
                mode = node.args[2]
                if mode not in ("circular", "replicate"):
                    continue

                x = node.args[0]
                pad = list(node.args[1])
                ndim = len(pad) // 2  # 1D: (l,r)  2D: (l,r,t,b)

                if mode == "circular":
                    new_val = self._insert_circular(g, x, pad, ndim, before=node)
                else:
                    new_val = self._insert_replicate(g, x, pad, ndim, before=node)

                self._copy_meta(node, new_val)
                node.replace_all_uses_with(new_val)
                g.erase_node(node)
                modified = True

        if modified:
            g.lint()
            gm.recompile()
        return PassResult(gm, modified)

    # ---------- rewrites ----------
    def _insert_circular(self, graph, x, pad, ndim, before):
        with graph.inserting_before(before):
            if ndim == 1:
                left, right = pad
                w = self._sym_size(graph, x, -1)
                start = self._sym_sub(graph, w, left)
                left_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, start, w))
                right_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, 0, right))
                self._copy_meta(x, left_slice)
                self._copy_meta(x, right_slice)
                out = graph.create_node("call_function", torch.ops.aten.cat.default, ((left_slice, x, right_slice), -1))
                self._copy_meta(x, out, lambda t: torch.cat([t[..., -left:], t, t[..., :right]], dim=-1))
                return out

            if ndim == 2:
                l, r, t, b = pad
                # horiz
                W = self._sym_size(graph, x, -1)
                start_w = self._sym_sub(graph, W, l)
                left_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, start_w, W))
                right_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, 0, r))
                self._copy_meta(x, left_slice)
                self._copy_meta(x, right_slice)
                x_cat = graph.create_node("call_function", torch.ops.aten.cat.default, ((left_slice, x, right_slice), -1))
                self._copy_meta(x, x_cat, lambda T: torch.cat([T[..., -l:], T, T[..., :r]], dim=-1))

                # vert
                H = self._sym_size(graph, x_cat, -2)
                start_h = self._sym_sub(graph, H, t)
                top_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x_cat, -2, start_h, H))
                bot_slice = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x_cat, -2, 0, b))
                self._copy_meta(x_cat, top_slice)
                self._copy_meta(x_cat, bot_slice)
                y_cat = graph.create_node("call_function", torch.ops.aten.cat.default, ((top_slice, x_cat, bot_slice), -2))
                self._copy_meta(x_cat, y_cat, lambda T: torch.cat([T[..., -t:, :], T, T[..., :b, :]], dim=-2))
                return y_cat

            raise NotImplementedError(f"circular pad only supports 1D/2D, got pad={pad}")

    def _insert_replicate(self, graph, x, pad, ndim, before):
        """
        Replicate: extend borders with edge values.
        Implemented via slice (edge 1-wide) + expand + cat.
        """
        with graph.inserting_before(before):
            if ndim == 1:
                left, right = pad
                parts = []
                if left > 0:
                    left_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, 0, 1))
                    self._copy_meta(x, left_edge)
                    left_pad = self._expand_along_dim(graph, left_edge, -1, left, before)
                    parts.append(left_pad)
                parts.append(x)
                if right > 0:
                    right_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, -1, None))
                    self._copy_meta(x, right_edge)
                    right_pad = self._expand_along_dim(graph, right_edge, -1, right, before)
                    parts.append(right_pad)

                out = parts[0] if len(parts) == 1 else graph.create_node("call_function", torch.ops.aten.cat.default, (tuple(parts), -1))
                # meta
                def _vt(t):
                    L = left; R = right
                    if L or R:
                        lp = t[..., :1].expand(*t.shape[:-1], L) if L else t[..., :0]
                        rp = t[..., -1:].expand(*t.shape[:-1], R) if R else t[..., :0]
                        return torch.cat([lp, t, rp], dim=-1)
                    return t
                self._copy_meta(x, out, _vt)
                return out

            if ndim == 2:
                l, r, t, b = pad
                # horizontal replicate first
                parts = []
                if l > 0:
                    left_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, 0, 1))
                    self._copy_meta(x, left_edge)
                    left_pad = self._expand_along_dim(graph, left_edge, -1, l, before)
                    parts.append(left_pad)
                parts.append(x)
                if r > 0:
                    right_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x, -1, -1, None))
                    self._copy_meta(x, right_edge)
                    right_pad = self._expand_along_dim(graph, right_edge, -1, r, before)
                    parts.append(right_pad)

                x_w = parts[0] if len(parts) == 1 else graph.create_node("call_function", torch.ops.aten.cat.default, (tuple(parts), -1))
                self._copy_meta(x, x_w, lambda T: torch.cat([
                    T[..., :1].expand(*T.shape[:-1], l) if l else T[..., :0],
                    T,
                    T[..., -1:].expand(*T.shape[:-1], r) if r else T[..., :0]
                ], dim=-1) if (l or r) else T)

                # then vertical replicate on the widened tensor
                parts2 = []
                if t > 0:
                    top_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x_w, -2, 0, 1))
                    self._copy_meta(x_w, top_edge)
                    top_pad = self._expand_along_dim(graph, top_edge, -2, t, before)
                    parts2.append(top_pad)
                parts2.append(x_w)
                if b > 0:
                    bot_edge = graph.create_node("call_function", torch.ops.aten.slice.Tensor, (x_w, -2, -1, None))
                    self._copy_meta(x_w, bot_edge)
                    bot_pad = self._expand_along_dim(graph, bot_edge, -2, b, before)
                    parts2.append(bot_pad)

                out = parts2[0] if len(parts2) == 1 else graph.create_node("call_function", torch.ops.aten.cat.default, (tuple(parts2), -2))
                self._copy_meta(x_w, out, lambda T: torch.cat([
                    T[..., :1, :].expand(*T.shape[:-2], t, T.shape[-1]) if t else T[..., :0, :],
                    T,
                    T[..., -1:, :].expand(*T.shape[:-2], b, T.shape[-1]) if b else T[..., :0, :]
                ], dim=-2) if (t or b) else T)
                return out

            raise NotImplementedError(f"replicate pad only supports 1D/2D, got pad={pad}")
