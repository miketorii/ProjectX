import networkx as nx
import matplotlib.pyplot as plt

print("=====================")

G = nx.Graph()

G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")

G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")

pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)

nx.draw(G, pos)

plt.savefig("exer6a.png")

print("=====================")

G.add_node("nodeD")
G.add_edge("nodeA","nodeD")
pos["nodeD"]=(1,0)

nx.draw(G, pos, with_labels=True )

plt.savefig("exer6a2.png")

print("=====================")
print("=====================")
