import networkx as nx


def path_node_to_path_tuple(path: list) -> list:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def get_outgoing_arc(nx_graph: nx.DiGraph, node: int) -> list[tuple]:
    return [(node, elem) for elem in nx_graph.neighbors(node)]


def get_encoded_outgoing_arc(nx_graph: nx.DiGraph, arc_dic: dict, node):
    return [arc_dic[elem] for elem in get_outgoing_arc(nx_graph, node)]


def get_incoming_arc(nx_graph: nx.DiGraph, node) -> list[tuple]:
    """
    Returns all arcs that get into the node "node".
    """
    return [(neighbor_node, node) for neighbor_node in nx_graph.neighbors(node)]


def get_encoded_incoming_arc(nx_graph: nx.DiGraph, arc_dic: dict, node: int) -> list[int]:
    """
    Returns all arcs that get into the node "node"
    and encode each arc as an integer take from arc_dic.
    """
    return [arc_dic[elem] for elem in get_incoming_arc(nx_graph, node)]
