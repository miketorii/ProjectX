from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import glob

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
