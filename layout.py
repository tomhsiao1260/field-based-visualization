import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    G = nx.Graph()

    G.add_node((0, 0))
    G.add_node((1, 1))
    G.add_edge((0, 0), (1, 1))

    G.add_node((0, 1))
    G.add_node((1, 0))
    G.add_edge((0, 1), (1, 0))

    data = np.full((4, 4), False)

    for node in G.nodes:
        data[node[0], node[1]] = True

    # G.remove_edge((0, 0), (1, 1))
    # G.remove_node((1, 1))

    print(G.number_of_nodes())
    print(G.number_of_edges())
    print(list(G.nodes))
    print(list(G.edges))
    print(list(G.adj[(0, 0)])) # G[(0, 0)]
    print(G.degree[(0, 0)])

    print(data)

    plt.imshow(data, cmap='gray')
    plt.axis("off") 
    plt.show()