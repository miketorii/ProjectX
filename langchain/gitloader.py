from langchain_community.document_loaders import GitLoader

print("----------------start-------------")

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./data",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loader.load()

print(len(raw_docs))

print("----------------end-------------")
