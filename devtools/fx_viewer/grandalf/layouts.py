# -*- coding: utf-8 -*-

"""
.. _layouts:

layouts.py
==========
Layouts are classes that provide graph drawing algorithms.

These classes all take a :class:`graph_core` argument. The graph
topology will never be permanently modified by the drawing algorithm:
e.g. "dummy" node insertion, edge reversal for making the graph
acyclic and so on, are all kept inside the layout object.
"""

# This code is part of Grandalf
# Copyright (C) 2010-2012 Axel Tillequin (bdcht3@gmail.com)
# published under the GPLv2 license or EPLv1 license

import importlib
from bisect import bisect
from sys import getrecursionlimit, setrecursionlimit

from .utils import *

# ------------------------------------------------------------------------------


class VertexViewer(object):
    """
    The VertexViewer class is used as the default provider of
    Vertex dimensions (w,h) and position (xy).
    In most cases it should be replaced by *view* instances associated
    with a ui widgets library, allowing to get dimensions and 
    set position directly on the widget.
    """

    def __init__(self, w=2, h=2, data=None):
        self.w = w
        self.h = h
        self.data = data
        self.xy = None

    def __str__(self, *args, **kwargs):
        return "VertexViewer (xy: %s) w: %s h: %s" % (self.xy, self.w, self.h)


# ------------------------------------------------------------------------------


class _sugiyama_vertex_attr(object):
    """
    The sugiyama layout adds new attributes to vertices.
    These attributes are stored in an internal _sugimyama_vertex_attr object.

    Attributes:
        rank (int): the rank number is the index of the layer that
                    contains this vertex.
        dummy (0/1): a flag indicating if the vertex is *dummy*
        pos (int): the index of the vertex in the layer
        x (list(float)): the list of computed horizontal coordinates of the vertex
        bar (float): the current *barycenter* of the vertex 
    """

    def __init__(self, r=None, d=0):
        self.rank = r
        self.dummy = d
        self.pos = None
        self.x = 0
        self.bar = None

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.rank, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s

    # def __eq__(self,x):
    #    return self.bar == x.bar
    # def __ne__(self,x):
    #    return self.bar != x.bar
    # def __lt__(self,x):
    #    return self.bar < x.bar
    # def __le__(self,x):
    #    return self.bar <= x.bar
    # def __gt__(self,x):
    #    return self.bar > x.bar
    # def __ge__(self,x):
    #    return self.bar >= x.bar


# ------------------------------------------------------------------------------


class DummyVertex(_sugiyama_vertex_attr):
    """
    The DummyVertex class is used by the sugiyama layout to represent
    *long* edges, i.e. edges that span over several ranks.
    For these edges, a DummyVertex is inserted in every inner layer.

    Attributes:
        view (viewclass): since a DummyVertex is acting as a Vertex, it
                          must have a view.
        ctrl (list[_sugiyama_attr]): the list of associated dummy vertices

    Methods:
        N(dir): reflect the Vertex method and returns the list of adjacent
                 vertices (possibly dummy) in the given direction.
        inner(dir): return True if a neighbor in the given direction is *dummy*.
    """

    def __init__(self, r=None, viewclass=VertexViewer):
        self.view = viewclass()
        self.ctrl = None
        super().__init__(r, d=1)

    def N(self, dir):
        assert dir == +1 or dir == -1
        v = self.ctrl.get(self.rank + dir, None)
        return [v] if v is not None else []

    def inner(self, dir):
        assert dir == +1 or dir == -1
        try:
            return any([x.dummy == 1 for x in self.N(dir)])
        except KeyError:
            return False
        except AttributeError:
            return False

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.rank, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s


# ------------------------------------------------------------------------------


class Layer(list):
    """
    Layer is where Sugiyama layout organises vertices in hierarchical lists.
    The placement of a vertex is done by the Sugiyama class, but it highly relies on
    the *ordering* of vertices in each layer to reduce crossings.
    This ordering depends on the neighbors found in the upper or lower layers.

    Attributes:
        layout (SugiyamaLayout): a reference to the sugiyama layout instance that
                                 contains this layer
        upper (Layer): a reference to the *upper* layer (rank-1)
        lower (Layer): a reference to the *lower* layer (rank+1)
        ccount (int) : number of crossings detected in this layer

    Methods:
        setup (layout): set initial attributes values from provided layout
        nextlayer(): returns *next* layer in the current layout's direction parameter.
        prevlayer(): returns *previous* layer in the current layout's direction parameter.
        order(): compute *optimal* ordering of vertices within the layer.
    """

    __r = None
    layout = None
    upper = None
    lower = None
    __x = 1.0
    ccount = None

    def __eq__(self, other):
        return super().__eq__(other)

    def __str__(self):
        s = "<Layer %d" % self.__r
        s += ", len=%d" % len(self)
        xc = self.ccount or "?"
        s += ", crossings=%s>" % xc
        return s

    def setup(self, layout):
        self.layout = layout
        r = layout.layers.index(self)
        self.__r = r
        if len(self) > 1:
            self.__x = 1.0 / (len(self) - 1)
        for i, v in enumerate(self):
            assert layout.grx[v].rank == r
            layout.grx[v].pos = i
            layout.grx[v].bar = i * self.__x
        if r > 0:
            self.upper = layout.layers[r - 1]
        if r < len(layout.layers) - 1:
            self.lower = layout.layers[r + 1]

    def nextlayer(self):
        return self.lower if self.layout.dirv == -1 else self.upper

    def prevlayer(self):
        return self.lower if self.layout.dirv == +1 else self.upper

    def order(self):
        sug = self.layout
        sug._edge_inverter()
        c = self._cc()
        if c > 0:
            for v in self:
                sug.grx[v].bar = self._meanvalueattr(v)
            # now resort layers l according to bar value:
            self.sort(key=lambda x: sug.grx[x].bar)
            # reduce & count crossings:
            c = self._ordering_reduce_crossings()
            # assign new position in layer l:
            for i, v in enumerate(self):
                sug.grx[v].pos = i
                sug.grx[v].bar = i * self.__x
        sug._edge_inverter()
        self.ccount = c
        return c

    def _meanvalueattr(self, v):
        """
        find new position of vertex v according to adjacency in prevlayer.
        position is given by the mean value of adjacent positions.
        experiments show that meanvalue heuristic performs better than median.
        """
        sug = self.layout
        if not self.prevlayer():
            return sug.grx[v].bar
        bars = [sug.grx[x].bar for x in self._neighbors(v)]
        return sug.grx[v].bar if len(bars) == 0 else float(sum(bars)) / len(bars)

    def _medianindex(self, v):
        """
        find new position of vertex v according to adjacency in layer l+dir.
        position is given by the median value of adjacent positions.
        median heuristic is proven to achieve at most 3 times the minimum
        of crossings (while barycenter achieve in theory the order of |V|)
        """
        assert self.prevlayer() != None
        N = self._neighbors(v)
        g = self.layout.grx
        pos = [g[x].pos for x in N]
        lp = len(pos)
        if lp == 0:
            return []
        pos.sort()
        pos = pos[:: self.layout.dirh]
        i, j = divmod(lp - 1, 2)
        return [pos[i]] if j == 0 else [pos[i], pos[i + j]]

    def _neighbors(self, v):
        """
        neighbors refer to upper/lower adjacent nodes.
        Note that v.N() provides neighbors of v in the graph, while
        this method provides the Vertex and DummyVertex adjacent to v in the
        upper or lower layer (depending on layout.dirv state).
        """
        assert self.layout.dag
        dirv = self.layout.dirv
        grxv = self.layout.grx[v]
        try:  # (cache)
            return grxv.nvs[dirv]
        except AttributeError:
            grxv.nvs = {-1: v.N(-1), +1: v.N(+1)}
            if grxv.dummy:
                return grxv.nvs[dirv]
            # v is real, v.N are graph neigbors but we need layers neighbors
            for d in (-1, +1):
                tr = grxv.rank + d
                for i, x in enumerate(v.N(d)):
                    if self.layout.grx[x].rank == tr:
                        continue
                    e = v.e_with(x)
                    dum = self.layout.ctrls[e][tr]
                    grxv.nvs[d][i] = dum
            return grxv.nvs[dirv]

    def _crossings(self):
        """
        counts (inefficently but at least accurately) the number of
        crossing edges between layer l and l+dirv.
        P[i][j] counts the number of crossings from j-th edge of vertex i.
        The total count of crossings is the sum of flattened P:
        x = sum(sum(P,[]))
        """
        g = self.layout.grx
        P = []
        for v in self:
            P.append([g[x].pos for x in self._neighbors(v)])
        for i, p in enumerate(P):
            candidates = sum(P[i + 1 :], [])
            for j, e in enumerate(p):
                p[j] = len(filter((lambda nx: nx < e), candidates))
            del candidates
        return P

    def _cc(self):
        """
        implementation of the efficient bilayer cross counting by insert-sort
        (see Barth & Mutzel paper "Simple and Efficient Bilayer Cross Counting")
        """
        g = self.layout.grx
        P = []
        for v in self:
            P.extend(sorted([g[x].pos for x in self._neighbors(v)]))
        # count inversions in P:
        s = []
        count = 0
        for i, p in enumerate(P):
            j = bisect(s, p)
            if j < i:
                count += i - j
            s.insert(j, p)
        return count

    def _ordering_reduce_crossings(self):
        assert self.layout.dag
        g = self.layout.grx
        N = len(self)
        X = 0
        for i, j in zip(range(N - 1), range(1, N)):
            vi = self[i]
            vj = self[j]
            ni = [g[v].bar for v in self._neighbors(vi)]
            Xij = Xji = 0
            for nj in [g[v].bar for v in self._neighbors(vj)]:
                x = len([nx for nx in ni if nx > nj])
                Xij += x
                Xji += len(ni) - x
            if Xji < Xij:
                self[i] = vj
                self[j] = vi
                X += Xji
            else:
                X += Xij
        return X


# ------------------------------------------------------------------------------


class SugiyamaLayout(object):
    """
    The Sugiyama layout is the traditional "layered" graph layout called
    *dot* in graphviz. This layout is quite efficient but heavily relies 
    on drawing heuristics. Adaptive drawing is limited to
    extending the leaves only, but since the algorithm is quite fast
    redrawing the entire graph (up to about a thousand nodes) gives
    usually good results in less than a second.

    The Sugiyama Layout Class takes as input a core_graph object and implements
    an efficient drawing algorithm based on nodes dimensions provided through
    a user-defined *view* property in each vertex.

    Attributes:
        dirvh (int): the current aligment state
                     for alignment policy:
                     dirvh=0 -> dirh=+1, dirv=-1: leftmost upper
                     dirvh=1 -> dirh=-1, dirv=-1: rightmost upper
                     dirvh=2 -> dirh=+1, dirv=+1: leftmost lower
                     dirvh=3 -> dirh=-1, dirv=+1: rightmost lower
        order_inter (int): the default number of layer placement iterations
        order_attr (str): set attribute name used for layer ordering
        xspace (int): horizontal space between vertices in a layer
        yspace (int): vertical space between layers
        dw (int): default width of a vertex
        dh (int): default height of a vertex
        g (graph_core): the graph component reference
        layers (list[Layer]): the list of layers
        grx (dict): associate vertex (possibly dummy) with their sugiyama attributes
        ctrls (dict): associate edge with all its vertices (including dummies)
        dag (bool): the current acyclic state 
        initdone (bool): True if state is initialized (see init_all).
    """

    def __init__(self, g):
        from .utils.geometry import median_wh

        # drawing parameters:
        self.dirvh = 0
        self.order_iter = 8
        self.order_attr = "pos"
        self.xspace = 20
        self.yspace = 20
        self.dw = 10
        self.dh = 10
        # For layered graphs, vertices and edges need to have some additional
        # attributes that make sense only for this kind of layout:
        # update graph struct:
        self.g = g
        self.layers = []
        self.grx = {}
        self.ctrls = {}
        self.dag = False
        for v in self.g.V():
            assert hasattr(v, "view")
            self.grx[v] = _sugiyama_vertex_attr()
        self.dw, self.dh = median_wh([v.view for v in self.g.V()])
        self.initdone = False

    def init_all(self, roots=None, inverted_edges=None, optimize=False):
        """initializes the layout algorithm by computing roots (unless provided),
           inverted edges (unless provided), vertices ranks and creates all dummy
           vertices and layers. 
             
             Parameters:
                roots (list[Vertex]): set *root* vertices (layer 0)
                inverted_edges (list[Edge]): set edges to invert to have a DAG.
                optimize (bool): optimize ranking if True (default False)
        """
        if self.initdone:
            return
        # For layered sugiyama algorithm, the input graph must be acyclic,
        # so we must provide a list of root nodes and a list of inverted edges.
        if roots is None:
            roots = [v for v in self.g.sV if len(v.e_in()) == 0]
        if inverted_edges is None:
            _ = self.g.get_scs_with_feedback(roots)
            inverted_edges = [x for x in self.g.sE if x.feedback]
        self.alt_e = inverted_edges
        # assign rank to all vertices:
        self.rank_all(roots, optimize)
        # add dummy vertex/edge for 'long' edges:
        for e in self.g.E():
            self.setdummies(e)
        # precompute some layers values:
        for l in self.layers:
            l.setup(self)
        self.initdone = True

    def draw(self, N=1.5):
        """compute every node coordinates after converging to optimal ordering by N
           rounds, and finally perform the edge routing.
        """
        while N > 0.5:
            for (l, mvmt) in self.ordering_step():
                pass
            N = N - 1
        if N > 0:
            for (l, mvmt) in self.ordering_step(oneway=True):
                pass
        self.setxy()
        self.draw_edges()

    def _edge_inverter(self):
        for e in self.alt_e:
            x, y = e.v
            e.v = (y, x)
        self.dag = not self.dag
        if self.dag:
            for e in self.g.degenerated_edges:
                e.detach()
                self.g.sE.remove(e)
        else:
            for e in self.g.degenerated_edges:
                self.g.add_edge(e)

    @property
    def dirvh(self):
        return self.__dirvh

    @property
    def dirv(self):
        return self.__dirv

    @property
    def dirh(self):
        return self.__dirh

    @dirvh.setter
    def dirvh(self, dirvh):
        assert dirvh in range(4)
        self.__dirvh = dirvh
        self.__dirh, self.__dirv = {0: (1, -1),
                                    1: (-1, -1),
                                    2: (1, 1),
                                    3: (-1, 1)}[dirvh]

    @dirv.setter
    def dirv(self, dirv):
        assert dirv in (-1, +1)
        dirvh = (dirv + 1) + (1 - self.__dirh) // 2
        self.dirvh = dirvh

    @dirh.setter
    def dirh(self, dirh):
        assert dirh in (-1, +1)
        dirvh = (self.__dirv + 1) + (1 - dirh) // 2
        self.dirvh = dirvh

    def rank_all(self, roots, optimize=False):
        """Computes rank of all vertices.
        add provided roots to rank 0 vertices,
        otherwise update ranking from provided roots.
        The initial rank is based on precedence relationships,
        optimal ranking may be derived from network flow (simplex).
        """
        self._edge_inverter()
        r = [x for x in self.g.sV if (len(x.e_in()) == 0 and x not in roots)]
        self._rank_init(roots + r)
        if optimize:
            self._rank_optimize()
        self._edge_inverter()

    def _rank_init(self, unranked):
        """Computes rank of provided unranked list of vertices and all
           their children. A vertex will be asign a rank when all its 
           inward edges have been *scanned*. When a vertex is asigned
           a rank, its outward edges are marked *scanned*.
        """
        assert self.dag
        scan = {}
        # set rank of unranked based on its in-edges vertices ranks:
        while len(unranked) > 0:
            l = []
            for v in unranked:
                self.setrank(v)
                # mark out-edges has scan-able:
                for e in v.e_out():
                    scan[e] = True
                # check if out-vertices are rank-able:
                for x in v.N(+1):
                    if not (False in [scan.get(e, False) for e in x.e_in()]):
                        if x not in l:
                            l.append(x)
            unranked = l

    def _rank_optimize(self):
        """optimize ranking by pushing long edges toward lower layers as much as possible.
        see other interersting network flow solver to minimize total edge length
        (http://jgaa.info/accepted/2005/EiglspergerSiebenhallerKaufmann2005.9.3.pdf)
        """
        assert self.dag
        for l in reversed(self.layers):
            for v in l:
                gv = self.grx[v]
                for x in v.N(-1):
                    if all((self.grx[y].rank >= gv.rank for y in x.N(+1))):
                        gx = self.grx[x]
                        self.layers[gx.rank].remove(x)
                        gx.rank = gv.rank - 1
                        self.layers[gv.rank - 1].append(x)

    def setrank(self, v):
        """set rank value for vertex v and add it to the corresponding layer.
           The Layer is created if it is the first vertex with this rank.
        """
        assert self.dag
        r = max([self.grx[x].rank for x in v.N(-1)] + [-1]) + 1
        self.grx[v].rank = r
        # add it to its layer:
        try:
            self.layers[r].append(v)
        except IndexError:
            assert r == len(self.layers)
            self.layers.append(Layer([v]))

    def dummyctrl(self, r, ctrl):
        """creates a DummyVertex at rank r inserted in the ctrl dict
           of the associated edge and layer.

           Arguments:
              r (int): rank value
              ctrl (dict): the edge's control vertices
           
           Returns:
              DummyVertex : the created DummyVertex.
        """
        dv = DummyVertex(r)
        dv.view.w, dv.view.h = self.dw, self.dh
        self.grx[dv] = dv
        dv.ctrl = ctrl
        ctrl[r] = dv
        self.layers[r].append(dv)
        return dv

    def setdummies(self, e):
        """creates and defines all needed dummy vertices for edge e.
        """
        v0, v1 = e.v
        r0, r1 = self.grx[v0].rank, self.grx[v1].rank
        if r0 > r1:
            assert e in self.alt_e
            v0, v1 = v1, v0
            r0, r1 = r1, r0
        if (r1 - r0) > 1:
            # "dummy vertices" are stored in the edge ctrl dict,
            # keyed by their rank in layers.
            ctrl = self.ctrls[e] = {}
            ctrl[r0] = v0
            ctrl[r1] = v1
            for r in range(r0 + 1, r1):
                self.dummyctrl(r, ctrl)

    def draw_step(self):
        """iterator that computes all vertices coordinates and edge routing after
           just one step (one layer after the other from top to bottom to top).
           Purely inefficient ! Use it only for "animation" or debugging purpose.
        """
        ostep = self.ordering_step()
        for s in ostep:
            self.setxy()
            self.draw_edges()
            yield s

    def ordering_step(self, oneway=False):
        """iterator that computes all vertices ordering in their layers
           (one layer after the other from top to bottom, to top again unless
           oneway is True).
        """
        self.dirv = -1
        crossings = 0
        for l in self.layers:
            mvmt = l.order()
            crossings += mvmt
            yield (l, mvmt)
        if oneway or (crossings == 0):
            return
        self.dirv = +1
        while l:
            mvmt = l.order()
            yield (l, mvmt)
            l = l.nextlayer()

    def setxy(self):
        """computes all vertex coordinates (x,y) using
        an algorithm by Brandes & Kopf.
        """
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float("infinity")
        # initialize vertex coordinates attributes:
        for l in self.layers:
            for v in l:
                self.grx[v].root = v
                self.grx[v].align = v
                self.grx[v].sink = v
                self.grx[v].shift = inf
                self.grx[v].X = None
                self.grx[v].x = [0.0] * 4
        curvh = self.dirvh  # save current dirvh value
        for dirvh in range(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = curvh  # restore it
        # vertical coordinate assigment of all nodes:
        Y = 0
        for l in self.layers:
            dY = max([v.view.h / 2.0 for v in l])
            for v in l:
                vx = sorted(self.grx[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1] + vx[2]) / 2.0
                # final xy-coordinates :
                v.view.xy = (avgm, Y + dY)
            
            # SORT by computed X to ensure correct order for overlap detection
            l.sort(key=lambda v: v.view.xy[0])
            
            # POST-PROCESSING: Fix any overlaps caused by BK variable-width averaging
            for i in range(1, len(l)):
                v_prev = l[i-1]
                v_curr = l[i]
                min_dist = self.xspace + (v_prev.view.w + v_curr.view.w) / 2.0
                if v_curr.view.xy[0] < v_prev.view.xy[0] + min_dist:
                    v_curr.view.xy = (v_prev.view.xy[0] + min_dist, v_curr.view.xy[1])
            
            Y += 2 * dY + self.yspace
        self._edge_inverter()

    def _detect_alignment_conflicts(self):
        """mark conflicts between edges:
        inner edges are edges between dummy nodes
        type 0 is regular crossing regular (or sharing vertex)
        type 1 is inner crossing regular (targeted crossings)
        type 2 is inner crossing inner (avoided by reduce_crossings phase)
        """
        curvh = self.dirvh  # save current dirvh value
        self.dirvh = 0
        self.conflicts = []
        for L in self.layers:
            last = len(L) - 1
            prev = L.prevlayer()
            if not prev:
                continue
            k0 = 0
            k1_init = len(prev) - 1
            l = 0
            for l1, v in enumerate(L):
                if not self.grx[v].dummy:
                    continue
                if l1 == last or v.inner(-1):
                    k1 = k1_init
                    if v.inner(-1):
                        k1 = self.grx[v.N(-1)[-1]].pos
                    for vl in L[l : l1 + 1]:
                        for vk in L._neighbors(vl):
                            k = self.grx[vk].pos
                            if k < k0 or k > k1:
                                self.conflicts.append((vk, vl))
                    l = l1 + 1
                    k0 = k1
        self.dirvh = curvh  # restore it

    def _coord_vertical_alignment(self):
        """performs vertical alignment according to current dirvh internal state.
        """
        dirh, dirv = self.dirh, self.dirv
        g = self.grx
        for l in self.layers[::-dirv]:
            if not l.prevlayer():
                continue
            r = None
            for vk in l[::dirh]:
                for m in l._medianindex(vk):
                    # take the median node in dirv layer:
                    um = l.prevlayer()[m]
                    # if vk is "free" align it with um's root
                    if g[vk].align is vk:
                        if dirv == 1:
                            vpair = (vk, um)
                        else:
                            vpair = (um, vk)
                        # if vk<->um link is used for alignment
                        if (vpair not in self.conflicts) and (
                            (r is None) or (dirh * r < dirh * m)
                        ):
                            g[um].align = vk
                            g[vk].root = g[um].root
                            g[vk].align = g[vk].root
                            r = m

    def _coord_horizontal_compact(self):
        limit = getrecursionlimit()
        N = len(self.layers) + 10
        if N > limit:
            setrecursionlimit(N)
        dirh, dirv = self.dirh, self.dirv
        g = self.grx
        L = self.layers[::-dirv]
        # recursive placement of blocks:
        for l in L:
            for v in l[::dirh]:
                if g[v].root is v:
                    self.__place_block(v)
        setrecursionlimit(limit)
        # mirror all nodes if right-aligned:
        if dirh == -1:
            for l in L:
                for v in l:
                    x = g[v].X
                    if x:
                        g[v].X = -x
        # then assign x-coord of its root:
        inf = float("infinity")
        rb = inf
        for l in L:
            for v in l[::dirh]:
                g[v].x[self.dirvh] = g[g[v].root].X
                rs = g[g[v].root].sink
                s = g[rs].shift
                if s < inf:
                    g[v].x[self.dirvh] += dirh * s
                rb = min(rb, g[v].x[self.dirvh])
        # normalize to 0, and reinit root/align/sink/shift/X
        for l in self.layers:
            for v in l:
                # g[v].x[dirvh] -= rb
                g[v].root = g[v].align = g[v].sink = v
                g[v].shift = inf
                g[v].X = None

    # TODO: rewrite in iterative form to avoid recursion limit...
    def __place_block(self, v):
        g = self.grx
        if g[v].X is None:
            # every block is initially placed at x=0
            g[v].X = 0.0
            # place block in which v belongs:
            w = v
            while 1:
                j = g[w].pos - self.dirh  # predecessor in rank must be placed
                r = g[w].rank
                if 0 <= j < len(self.layers[r]):
                    wprec = self.layers[r][j]
                    delta = (
                        self.xspace + (wprec.view.w + w.view.w) / 2.0
                    )  # abs positive minimum displ.
                    # take root and place block:
                    u = g[wprec].root
                    self.__place_block(u)
                    # set sink as sink of prec-block root
                    if g[v].sink is v:
                        g[v].sink = g[u].sink
                    if g[v].sink != g[u].sink:
                        s = g[u].sink
                        newshift = g[v].X - (g[u].X + delta)
                        g[s].shift = min(g[s].shift, newshift)
                    else:
                        g[v].X = max(g[v].X, (g[u].X + delta))
                # take next node to align in block:
                w = g[w].align
                # quit if self aligned
                if w is v:
                    break

    def draw_edges(self):
        """Basic edge routing applied only for edges with dummy points.
        Enhanced edge routing can be performed by using the apropriate
        *route_with_xxx* functions from :ref:routing_ in the edges' view.
        """
        for e in self.g.E():
            if hasattr(e, "view"):
                l = []
                if e in self.ctrls:
                    D = self.ctrls[e]
                    r0, r1 = self.grx[e.v[0]].rank, self.grx[e.v[1]].rank
                    if r0 < r1:
                        ranks = range(r0 + 1, r1)
                    else:
                        ranks = range(r0 - 1, r1, -1)
                    l = [D[r].view.xy for r in ranks]
                l.insert(0, e.v[0].view.xy)
                l.append(e.v[1].view.xy)
                try:
                    self.route_edge(e, l)
                except AttributeError:
                    pass
                e.view.setpath(l)


#  DIRECTED GRAPH WITH CONSTRAINTS LAYOUT
# ------------------------------------------------------------------------------


class DigcoLayout(object):
    linalg = importlib.import_module(__package__ + ".utils.geometry")

    def __init__(self, g):
        # drawing parameters:
        self.xspace = 10
        self.yspace = 10
        self.dr = 10
        self.debug = False

        self.g = g
        self.levels = []
        for i, v in enumerate(self.g.V()):
            assert hasattr(v, "view")
            v.i = i
            self.dr = max((self.dr, v.view.w, v.view.h))
        # solver parameters:
        self._cg_max_iter = g.order()
        self._cg_tolerance = 1.0e-6
        self._eps = 1.0e-5
        self._cv_max_iter = self._cg_max_iter

    def init_all(self, alpha=0.1, beta=0.01):
        y = None
        if self.g.directed:
            # partition g in hierarchical levels:
            y = self.part_to_levels(alpha, beta)
        # initiate positions (y and random in x):
        self.Z = self._xyinit(y)

    def draw(self, N=None):
        if N is None:
            N = self._cv_max_iter
        self.Z = self._optimize(self.Z, limit=N)
        # set view xy from near-optimal coords matrix:
        for v in self.g.V():
            v.view.xy = (self.Z[v.i][0, 0] * self.dr, self.Z[v.i][0, 1] * self.dr)
        self.draw_edges()

    def draw_step(self):
        for x in range(self._cv_max_iter):
            self.draw(N=1)
            self.draw_edges()
            yield

    # Basic edge routing with segments
    def draw_edges(self):
        for e in self.g.E():
            if hasattr(e, "view"):
                l = [e.v[0].view.xy, e.v[1].view.xy]
                try:
                    self.route_edge(e, l)
                except AttributeError:
                    pass
                e.view.setpath(l)

    # partition the nodes into levels:
    def part_to_levels(self, alpha, beta):
        opty, err = self.optimal_arrangement()
        ordering = list(zip(opty, self.g.sV))
        eps = alpha * (opty.max() - opty.min()) / (len(opty) - 1)
        eps = max(beta, eps)
        sorted(ordering, reverse=True)
        l = []
        self.levels.append(l)
        for i in range(len(list(ordering)) - 1):
            y, v = ordering[i]
            l.append(v)
            v.level = self.levels.index(l)
            if (y - ordering[i + 1][0]) > eps:
                l = []
                self.levels.append(l)
        y, v = ordering[-1]
        l.append(v)
        v.level = self.levels.index(l)
        return opty

    def optimal_arrangement(self):
        b = self.balance()
        y = DigcoLayout.linalg.rand_ortho1(self.g.order())
        return self._conjugate_gradient_L(y, b)

    # balance vector is assembled in finite-element way...
    # this is faster than computing b[i] for each i.
    def balance(self):
        b = DigcoLayout.linalg.array([0.0] * self.g.order(), dtype=float)
        for e in self.g.E():
            s = e.v[0]
            d = e.v[1]
            q = e.w * (self.yspace + (s.view.h + d.view.h) / 2.0)
            b[s.i] += q
            b[d.i] -= q
        return b

    # We compute the solution Y of L.Y = b by conjugate gradient method
    # (L is semi-definite positive so Y is unique and convergence is O(n))
    # note that only arrays are involved here...
    def _conjugate_gradient_L(self, y, b):
        Lii = self.__Lii_()
        r = b - self.__L_pk(Lii, y)
        p = DigcoLayout.linalg.array(r, copy=True)
        rr = sum(r * r)
        for k in range(self._cg_max_iter):
            try:
                Lp = self.__L_pk(Lii, p)
                alpha = rr / sum(p * Lp)
                y += alpha / p
                r -= alpha * Lp
                newrr = sum(r * r)
                beta = newrr / rr
                rr = newrr
                if rr < self._cg_tolerance:
                    break
                p = r + beta * p
            except ZeroDivisionError:
                return (None, rr)
        return (y, rr)

    # _xyinit can use diagonally scaled initial vertices positioning to provide
    # better convergence in constrained stress majorization
    def _xyinit(self, y=None):
        if y is None:
            y = DigcoLayout.linalg.rand_ortho1(self.g.order())
        x = DigcoLayout.linalg.rand_ortho1(self.g.order())
        # translate and normalize:
        x = x - x[0]
        y = y - y[0]
        sfactor = 1.0 / max(list(map(abs, y)) + list(map(abs, x)))
        return DigcoLayout.linalg.matrix(list(zip(x * sfactor, y * sfactor)))

    # provide the diagonal of the Laplacian matrix of g
    # the rest of L (sparse!) is already stored in every edges.
    def __Lii_(self):
        Lii = []
        for v in self.g.V():
            Lii.append(sum([e.w for e in v.e]))
        return DigcoLayout.linalg.array(Lii, dtype=float)

    # we don't compute the L.Pk matrix/vector product here since
    # L is sparse (order of |E| not |V|^2 !) so we let each edge
    # contribute to the resulting L.Pk vector in a FE assembly way...
    def __L_pk(self, Lii, pk):
        y = Lii * pk
        for e in self.g.sE:
            i1 = e.v[0].i
            i2 = e.v[1].i
            y[i1] -= e.w * pk[i2]
            y[i2] -= e.w * pk[i1]
        return y

    # conjugate_gradient with given matrix Lw:
    # it is assumed that b is not a multivector,
    # so _cg_Lw should be called in all directions separately.
    # note that everything is a matrix here, (arrays are row vectors only)
    def _cg_Lw(self, Lw, z, b):
        scal = lambda U, V: float(U.transpose() * V)
        r = b - Lw * z
        p = r.copy()
        rr = scal(r, r)
        for k in range(self._cg_max_iter):
            if rr < self._cg_tolerance:
                break
            Lp = Lw * p
            alpha = rr / scal(p, Lp)
            z = z + alpha * p
            r = r - alpha * Lp
            newrr = scal(r, r)
            beta = newrr / rr
            rr = newrr
            p = r + beta * p
        return (z, rr)

    def __Dij_(self):
        Dji = []
        for v in self.g.V():
            wd = self.g.dijkstra(v)
            Di = [wd[w] for w in self.g.V()]
            Dji.append(Di)
        # at this point  D is stored by rows,
        # but anymway it's a symmetric matrix
        return DigcoLayout.linalg.matrix(Dji, dtype=float)

    # returns matrix -L^w
    def __Lij_w_(self):
        self.Dij = self.__Dij_()  # we keep D also for L^Z computations
        Lij = self.Dij.copy()
        n = self.g.order()
        for i in range(n):
            d = 0
            for j in range(n):
                if j == i:
                    continue
                Lij[i, j] = 1.0 / self.Dij[i, j] ** 2
                d += Lij[i, j]
            Lij[i, i] = -d
        return Lij

    # returns vector -L^Z.Z:
    def __Lij_Z_Z(self, Z):
        n = self.g.order()
        # init:
        lzz = Z.copy() * 0.0  # lzz has dim Z (n x 2)
        liz = DigcoLayout.linalg.matrix([0.0] * n)  # liz is a row of L^Z (size n)
        # compute lzz = L^Z.Z while assembling L^Z by row (liz):
        for i in range(n):
            iterk_except_i = (k for k in range(n) if k != i)
            for k in iterk_except_i:
                v = Z[i] - Z[k]
                liz[0, k] = 1.0 / (
                    self.Dij[i, k] * DigcoLayout.linalg.sqrt(v * v.transpose())
                )
            liz[0, i] = 0.0  # forced, otherwise next liz.sum() is wrong !
            liz[0, i] = -liz.sum()
            # now that we have the i-th row of L^Z, just dotprod with Z:
            lzz[i] = liz * Z
        return lzz

    def _optimize(self, Z, limit=100):
        Lw = self.__Lij_w_()
        K = self.g.order() * (self.g.order() - 1.0) / 2.0
        stress = float("inf")
        count = 0
        deep = 0
        b = self.__Lij_Z_Z(Z)
        while count < limit:
            if self.debug:
                print("count %d" % count)
                print("Z = ", Z)
                print("b = ", b)
            # find next Z by solving Lw.Z = b in every direction:
            x, xerr = self._cg_Lw(Lw[1:, 1:], Z[1:, 0], b[1:, 0])
            y, yerr = self._cg_Lw(Lw[1:, 1:], Z[1:, 1], b[1:, 1])
            Z[1:, 0] = x
            Z[1:, 1] = y
            if self.debug:
                print(" cg -> ")
                print(Z, xerr, yerr)
            # compute new stress:
            FZ = K - float(x.transpose() * b[1:, 0] + y.transpose() * b[1:, 1])
            # precompute new b:
            b = self.__Lij_Z_Z(Z)
            # update new stress:
            FZ += 2 * float(x.transpose() * b[1:, 0] + y.transpose() * b[1:, 1])
            # test convergence:
            print("stress=%.10f" % FZ)
            if stress == 0.0:
                break
            elif abs((stress - FZ) / stress) < self._eps:
                if deep == 2:
                    break
                else:
                    deep += 1
            stress = FZ
            count += 1
        return Z


# ------------------------------------------------------------------------------
class DwyerLayout(DigcoLayout):
    pass
