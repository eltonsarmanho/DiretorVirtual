import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatMaritalk
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts.chat import ChatPromptTemplate

load_dotenv()

def resposta(pergunta):
    llm = ChatMaritalk(
        model="sabia-3",
        api_key= os.getenv("CHAVE_API"),
        temperature=0.7,
        #max_tokens=150
    )

    pdf_folder_path = '/home/augustinho/PycharmProjects/AssistenteVirtual/files'
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            #print(f'Arquivo: {loader}')
            documents.extend(loader.load())

    # loader = PyPDFLoader('/home/augustinho/PycharmProjects/AssistenteVirtual/files/Disciplinas.pdf')
    # documents = loader.load()
    #print(f'\n\nDocumento: {documents}')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    retriever = BM25Retriever.from_documents(texts)

    prompt = """
        Utilze apenas os documentos para responder as perguntas, caso contrário, retorne a seguinte resposta: "Não conseguimos responder sua pergunta, por favor, entre em contato com a secretaria.".
        {context}

        Pergunta: {query}
    """

    prompt_template = ChatPromptTemplate.from_messages([("human", prompt)])
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=prompt_template)

    query = pergunta

    docs = retriever.invoke(query)

    response = chain.invoke(
        {"input_documents": docs, "query": query}
    )


    return response['output_text']

#print(resposta('qual o tempo de duração do curso de sistemas de informação?'))

# print(resposta('o que é o Trabalho de Conclusão de Curso?'))
# print(resposta('Qual a carga horária de disciplinas obrigatórias?'))
# print(resposta('Qual o total de cargahorŕia do curso de Sistemas de Informação?'))
#print(resposta('Qual as disciplinas do primeiro semestre ou período do curso?'))