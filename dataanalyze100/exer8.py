import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

df_links = pd.read_csv("./data/links.csv", index_col="Node")
print( df_links.head() )

G = nx.Graph()

NUM = len(df_links.index)
for i in range(NUM):
    node_no = df_links.columns[i].strip("Node")
    G.add_node(str(node_no))

for i in range(NUM):
    for j in range(NUM):
        node_name = "Node" + str(j)
        if df_links[node_name].iloc[i]==1:
            G.add_edge( str(i), str(j))

nx.draw_networkx(G, node_color="k", edge_color="k", font_color="w")

plt.savefig("exer8.png")

print("----------------------------------------")





