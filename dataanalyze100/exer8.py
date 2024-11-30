import pandas as pd
import numpy as np

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

def determin_link(percent):
    rand_val = np.random.rand()
    if rand_val <= percent:
        return 1
    else:
        return 0

def simulate_percolation(num, list_active, percent_percolation):
    for i in range(num):
        if list_active[i]==1:
            for j in range(num):
                node_name = "Node" + str(j)
                if df_links[node_name].iloc[i]==1:
                    if determin_link(percent_percolation)==1:
                        list_active[j] = 1
    return list_active

percent_percolation=0.1
T_NUM = 36
NUM = len(df_links.index)
list_active = np.zeros(NUM)
list_active[0] = 1

list_timeSeries = []

for t in range(T_NUM):
    list_active = simulate_percolation(NUM, list_active, percent_percolation)
    list_timeSeries.append(list_active.copy())

print( list_timeSeries )

def active_node_coloring(t, list_active):
    print(list_timeSeries[t])
    list_color = []
    for i in range( len(list_timeSeries[t]) ):
        if list_timeSeries[t][i]==1:
            list_color.append("r")
        else:
            list_color.append("k")
    return list_color

print("----------------------------------------")

t = 0
nx.draw_networkx(G, font_color="w",node_color=active_node_coloring(t,list_timeSeries[t]) )
plt.savefig("exer8t0.png")

t = 11
nx.draw_networkx(G, font_color="w",node_color=active_node_coloring(t,list_timeSeries[t]) )
plt.savefig("exer8t11.png")

t = 35
nx.draw_networkx(G, font_color="w",node_color=active_node_coloring(t,list_timeSeries[t]) )
plt.savefig("exer8t35.png")

print("----------------------------------------")





