from importlib.metadata import version

pkgs = [
    "matplotlib",
    "numpy",
    "tiktoken",
    "torch",
    "tensorflow",
    "pandas",        
]

for p in pkgs:
    print(f"{p} version: {version(p)}")
