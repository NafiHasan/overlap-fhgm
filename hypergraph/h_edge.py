"""HyperEdge data structure for labeled hyperedges."""


class HyperEdge:
    """
    A labeled hyperedge connecting an arbitrary set of vertices.

    Attributes:
        id (int): Unique identifier for this hyperedge within a hypergraph.
        vertices (list): List of vertex IDs that this hyperedge connects.
        label: Label assigned to this hyperedge.
    """

    def __init__(self, id, vertices, label):
        self.id = id
        self.vertices = vertices
        self.label = label
