# This code is part of Grandalf
#  Copyright (C) 2014 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Fabio Zadrozny

__all__ = [
    "convert_grandalf_graph_to_networkx_graph",
    "convert_nextworkx_graph_to_grandalf",
]

# Some utilities to interact with networkx.

# Converts a grandalf graph to a networkx graph.
# Note that the edge concept is the same, but a vertex in grandalf is called a node in networkx.
def convert_grandalf_graph_to_networkx_graph(G):
    from networkx import MultiDiGraph

    nxg = MultiDiGraph()
    for v in G.V():
        nxg.add_node(v.data)
    for e in G.E():
        nxg.add_edge(e.v[0].data, e.v[1].data)
    return nxg


# Converts a networkx graph to a grandalf graph.
# Note that the edge concept is the same, but a vertex in grandalf is called a node in networkx.
def convert_nextworkx_graph_to_grandalf(G):
    from ..graphs import Graph, Vertex, Edge

    V = []
    data_to_V = {}
    for x in G.nodes():
        vertex = Vertex(x)
        V.append(vertex)
        data_to_V[x] = vertex
    E = [Edge(data_to_V[xy[0]], data_to_V[xy[1]], data=xy) for xy in G.edges()]
    g = Graph(V, E)
    return g
