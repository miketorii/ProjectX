import uuid
import faiss
import numpy as np

print("----------------------------------------------------")
print("--------------------Test IndexFlatL2----------------")
print("----------------------------------------------------")

uuidbin = uuid.uuid4()
print(uuidbin)

d = 128 # numbers of vector dimension
myindex = faiss.IndexFlatL2(d)
print(myindex)

nb = 1000 # numbers of vector
np.random.seed(123)
vectors = np.random.random((nb, d)).astype('float32')
myindex.add(vectors)

nq = 10
queries = np.random.random((nq, d)).astype('float32')
print("-------------------queries-----------------------")
print(queries)

k = 5
distances, indices = myindex.search(queries, k)
print("--------------------distances----------------------")
print(distances)
print("--------------------indices----------------------")
print(indices)

print("----------------------------------------------------")
print("--------------------dictionary----------------------")
print("----------------------------------------------------")

my_dict = {"key1":10, "key2":20, "key3":30}
print(my_dict["key1"])

print(my_dict.keys())
print(my_dict.values())
print(my_dict.items())

del my_dict["key2"]
print(my_dict.items())

squares = {x: x*x for x in range(5)}
print(squares)

print("----------------------------------------------------")
print("--------------------zip-----------------------------")
print("----------------------------------------------------")

keys = ["key1", "key2", "key3"]
values = [11,12,13]
print(keys)
print(values)

for num, char in zip(keys, values):
    print(f"{num} {char}")

print("--------")    
my_dict = dict(zip(keys,values))
print(my_dict)
