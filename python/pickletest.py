import pickle

data = {"key": "value", "list": [1,2,3]}

with open("data.pkl","wb") as f:
    pickle.dump(data, f)

with open("data.pkl","rb") as f:
    loaded_data = pickle.load(f)

print(loaded_data)
