import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("==========================================")

df_w = pd.read_csv("./data/network_weight.csv")
df_p = pd.read_csv("./data/network_pos.csv")

print( df_w.head() )
print( len(df_w) )

print( df_p.head() )
print( len(df_p) )

G = nx.Graph()

for i in range( len(df_w.columns) ):
    G.add_node(df_w.columns[i])

size = 10
edge_weights = []

for i in range( len(df_w.columns) ):
    for j in range( len(df_w.columns) ):
        if not (i==j):
            G.add_edge( df_w.columns[i], df_w.columns[j] )
            edge_weights.append( df_w.iloc[i].iloc[j]*size )

pos={}
for i in range( len(df_w.columns) ):
    node = df_w.columns[i]
    pos[node] = ( df_p[node][0], df_p[node][1] )

nx.draw(G, pos, with_labels=True, font_size=16 , node_size=1000, node_color="k", font_color="w", width=edge_weights )

plt.savefig("exer6b.png")

print("==========================================")
