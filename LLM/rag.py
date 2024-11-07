import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever


def retriever():

    pdf_folder_path = '/home/augustinho/PycharmProjects/AssistenteVirtual/files'
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    # loader1 = PyPDFLoader(file_path='files/Cartilha_de_Estágio')
    #
    # data = loader1.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    retriever = BM25Retriever.from_documents(texts)

    return retriever
