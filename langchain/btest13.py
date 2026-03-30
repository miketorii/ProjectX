from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import glob

all_documents = []

loader = PyPDFLoader("data/test.pdf")
pages = loader.load_and_split()

print(pages[0])

