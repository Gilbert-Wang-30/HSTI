import logging
from py2neo import Graph
import networkx as nx
import json

def load_sfg_from_neo4j(uri, user, password):
    """
    Connect to a Neo4j instance and load the Scenario Feature Graph (SFG)
    into a NetworkX MultiDiGraph.
    """
    logger = logging.getLogger(__name__)
    logger.info("Connecting to Neo4j at %s as user '%s'...", uri, user)
    graph_db = Graph(uri, auth=(user, password))
    logger.info("Connection successful. Retrieving nodes and relationships...")

    G = nx.MultiDiGraph()

    # Load DataSlice nodes
    ds_nodes = graph_db.run("MATCH (n:DataSlice) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props").data()
    for rec in ds_nodes:
        node_id = rec['id']
        props = rec['props'] or {}
        props['labels'] = rec['labels']
        G.add_node(node_id, **props)

    # Load Feature nodes
    ft_nodes = graph_db.run("MATCH (n:Feature) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props").data()
    for rec in ft_nodes:
        node_id = rec['id']
        props = rec['props'] or {}
        props['labels'] = rec['labels']
        G.add_node(node_id, **props)

    # Load CORRELATION edges
    corr_edges = graph_db.run("MATCH (a:DataSlice)-[r:CORRELATION]->(b:DataSlice) RETURN id(a) AS src, id(b) AS dst, r.weight AS weight").data()
    for rec in corr_edges:
        G.add_edge(rec['src'], rec['dst'], type='CORRELATION', weight=rec.get('weight', 1.0))

    # Load AFFILIATION edges
    aff_edges = graph_db.run("MATCH (a:DataSlice)-[r:AFFILIATION]->(b:Feature) RETURN id(a) AS src, id(b) AS dst, r.weight AS weight").data()
    for rec in aff_edges:
        G.add_edge(rec['src'], rec['dst'], type='AFFILIATION', weight=rec.get('weight', 1.0))

    # Load CAUSALITY edges
    caus_edges = graph_db.run("MATCH (a:Feature)-[r:CAUSALITY]->(b:Feature) RETURN id(a) AS src, id(b) AS dst, r.weight AS weight").data()
    for rec in caus_edges:
        G.add_edge(rec['src'], rec['dst'], type='CAUSALITY', weight=rec.get('weight', 1.0))

    logger.info("Loaded %d nodes and %d edges into NetworkX graph.", G.number_of_nodes(), G.number_of_edges())
    return G

def print_sfg_info(G):
    """
    Print basic information about the loaded SFG.
    """
    print("SFG Information:")
    print("Loaded {} nodes:".format(G.number_of_nodes()))
    print("  DataSlice:", sum('DataSlice' in d.get('labels', []) for n, d in G.nodes(data=True)))
    print("  Feature:  ", sum('Feature' in d.get('labels', []) for n, d in G.nodes(data=True)))
    print("Loaded {} edges:".format(G.number_of_edges()))
    print("  CORRELATION:", sum(d.get('type') == 'CORRELATION' for u, v, d in G.edges(data=True)))
    print("  AFFILIATION:", sum(d.get('type') == 'AFFILIATION' for u, v, d in G.edges(data=True)))
    print("  CAUSALITY:  ", sum(d.get('type') == 'CAUSALITY' for u, v, d in G.edges(data=True)))

def sfg_to_pstgs(sfg):
    """
    Partition SFG DataSlice nodes into PSTGs by correlation-connected components.
    Each PSTG contains a set of correlated DataSlice nodes.
    Features and other edge types are ignored for now.
    """
    pstgs = []
    # Extract all DataSlice nodes
    dataslice_nodes = [n for n, d in sfg.nodes(data=True) if 'DataSlice' in d.get('labels', [])]

    # Build correlation-only undirected subgraph for grouping
    G_corr = nx.Graph()
    for u, v, d in sfg.edges(data=True):
        if d.get('type') == 'CORRELATION' and u in dataslice_nodes and v in dataslice_nodes:
            G_corr.add_edge(u, v)
    G_corr.add_nodes_from(dataslice_nodes)  # in case isolated nodes exist

    # Identify connected components (each is a PSTG group)
    for i, component in enumerate(nx.connected_components(G_corr)):
        pstg = sfg.subgraph(component).copy()
        pstgs.append(pstg)
        with open(f"pstg_group_{i}.json", "w") as f:
            json.dump(nx.node_link_data(pstg), f, indent=2)

    print(f"Converted SFG to {len(pstgs)} PSTGs.")
    return pstgs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    G = load_sfg_from_neo4j(uri, user, password)
    print_sfg_info(G)
    print("SFG loaded with {} nodes and {} edges.".format(G.number_of_nodes(), G.number_of_edges()))
    pstgs = sfg_to_pstgs(G)
    