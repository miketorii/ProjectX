from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import glob

from langchain.text_splitter import CharacterTextSplitter

all_documents = []

loader = PyPDFLoader("data/test.pdf")
pages = loader.load_and_split()

print(pages[0])

csv_files = glob.glob("data/*.csv")

csv_files = [f for f in csv_files if "test" in f]

for csv_file in csv_files:
    loader = CSVLoader(file_path=csv_file)
    data = loader.load()
    all_documents.extend(data)

print(all_documents)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=0
)

urls = [
    "https://storage.googleapis.com/oreilly-content/NutriFusion%20Foods%20Marketing%20Plan%202022.docx",
    "https://storage.googleapis.com/oreilly-content/NutriFusion%20Foods%20Marketing%20Plan%202023.docx",
]

docs = []

for url in urls:
    loader = Docx2txtLoader(url)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = "NutriFusion Foods Marketing Plan - 2022/2023"
    docs.extend(chunks)

all_documents.extend(docs)

print(all_documents)
