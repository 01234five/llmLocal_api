import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
import Constants as _constants
import GlobalFunctions as global_functions


import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List
from Constants import (
    EMBEDDING_MODEL_NAME
)
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv'] ]




def main(_device_type,source_directory):
    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text")
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    db = global_functions.DatabaseReturn(_constants.PERSIST_DIRECTORY,HuggingFaceInstructEmbeddings(model_name=_constants.EMBEDDING_MODEL_NAME, model_kwargs={"device": _device_type}),_constants.CHROMA_SETTINGS) 
    # Add the new documents to the existing Chroma database
    db.add_documents(texts)
    db.persist()
    db = None
    print("Completed.")
if __name__ == "__main__":
    device_type=""
    if torch.cuda.is_available():
        device_type="cuda"  
    else: 
        device_type="cpu",
    _rd=os.path.dirname(os.path.realpath(__file__))
    _sd=f"{_rd}/documents/document1/"
    main(device_type,_sd)