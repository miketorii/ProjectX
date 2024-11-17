import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df_tr = pd.read_csv("./data/trans_route.csv", index_col="工場")
print( df_tr.head() )
df_pos = pd.read_csv("./data/trans_route_pos.csv")
print( df_pos.head() )

G = nx.Graph()

for i in range( len(df_pos.columns) ):
    G.add_node(df_pos.columns[i])

num_pre = 0
edge_weights = []
size = 0.1

for i in range( len(df_pos.columns) ):
    for j in range( len(df_pos.columns) ):
        if not (i==j):
            G.add_edge( df_pos.columns[i], df_pos.columns[j] )

            if num_pre < len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                
                if ( df_pos.columns[i] in df_tr.columns) and ( df_pos.columns[j] in df_tr.index):
                    if df_tr[ df_pos.columns[i] ][ df_pos.columns[j] ]:
                        weight = df_tr[ df_pos.columns[i] ][ df_pos.columns[j] ]*size

                elif ( df_pos.columns[j] in df_tr.columns) and ( df_pos.columns[i] in df_tr.index):
                    if df_tr[ df_pos.columns[j] ][ df_pos.columns[i] ]:
                        weight = df_tr[ df_pos.columns[j] ][ df_pos.columns[i] ]*size
                        
                edge_weights.append( weight )
                
pos={}
for i in range( len(df_pos.columns) ):
    node = df_pos.columns[i]
    pos[node] = ( df_pos[node][0], df_pos[node][1] )

nx.draw(G, pos, with_labels=True, font_size=16 , node_size=1000, node_color="k", font_color="w", width=edge_weights )

plt.savefig("exer6c.png")

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")




