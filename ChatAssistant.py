import os
import json
import re
import pickle
from dotenv import load_dotenv
from langchain_community.chat_models import ChatMaritalk
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts.chat import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor


class ChatAssistant:
    def __init__(self, config_path='Keys/config.json', pdf_folder_path='files'):
        load_dotenv()
        self.config = self._load_config(config_path)
        self.llm = self._initialize_llm()
        self.texts = self._load_documents(pdf_folder_path)
        self.retriever = BM25Retriever.from_documents(self.texts, top_k=10)
        self.prompt_template = self._create_prompt_template()
        self.chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True, prompt=self.prompt_template)

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("Arquivo de configuração JSON não encontrado.")
        except json.JSONDecodeError:
            raise ValueError("Erro ao decodificar o arquivo JSON.")

    def _initialize_llm(self):
        minha_chave = self.config.get('minha_chave')
        return ChatMaritalk(
            model="sabia-3",
            api_key=minha_chave,
            temperature=0.3,
            max_tokens=100
        )

    def _load_documents(self, pdf_folder_path):
        cache_file = "document_cache.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                return pickle.load(file)

        documents = []
        pdf_paths = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path) if
                     file.endswith('.pdf')]

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda path: PyPDFLoader(path).load(), pdf_paths)
            for result in results:
                documents.extend(result)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n", " ", ""])
        texts = text_splitter.split_documents(documents)

        with open(cache_file, "wb") as file:
            pickle.dump(texts, file)

        return texts

    def _create_prompt_template(self):
        prompt = """
            Utilize apenas os documentos para responder as perguntas, caso contrário, retorne a seguinte resposta: 
            "Não conseguimos responder sua pergunta, por favor, entre em contato com a secretaria.".
            {context}

            Pergunta: {query}
        """
        return ChatPromptTemplate.from_messages([("human", prompt)])

    def _normalize_query(self, query):
        ordinals = {
            "primeiro": "1º", "um": "1º","1°":"1º",
            "segundo": "2º", "dois": "2º","2°":"2º",
            "terceiro": "3º", "três": "3º","3°":"3º",
            "quarto": "4º", "quatro": "4º","4°":"4º",
            "quinto": "5º", "cinco": "5º","5°":"5º",
            "sexto": "6º", "seis": "6º","6°":"6º",
            "sétimo": "7º", "sete": "7º","7°":"7º",
            "oitavo": "8º", "oito": "8º","8°":"8º",

        }

        synonyms = {
            "semestre": "semestre",
            "bloco": "semestre",
            "etapa": "semestre",
            "fase": "semestre"
        }

        for word, num in ordinals.items():
            query = re.sub(rf"\b{word}\b", num, query, flags=re.IGNORECASE)

        for synonym, standard in synonyms.items():
            query = re.sub(rf"\b{synonym}\b", standard, query, flags=re.IGNORECASE)

        return query

    def get_response(self, pergunta):
        pergunta = self._normalize_query(pergunta)
        docs = self.retriever.invoke(pergunta)

        response = self.chain.invoke({"input_documents": docs, "query": pergunta})
        return response.get('output_text', "Erro ao gerar resposta")

# Exemplo de uso da classe
# assistant = ChatAssistant()
# print(assistant.get_response('Quais as disciplinas do sexto semestre?'))
# print(assistant.get_response('Qual a carga horária de Estágio?'))
# print(assistant.get_response('Qual a carga horária da disciplina ESTÁGIO II?'))
# print(assistant.get_response('As disciplinas do 7º semestre são?'))
# print(assistant.get_response('Qual conteúdo da disciplina de ESTRUTURAS DE DADOS II?'))