# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains.question_answering import load_qa_chain
# from langchain_core.prompts.chat import ChatPromptTemplate
#
# from llm import llm
# from src.rag import retriever
#
# def resposta_duvida(pergunta):
#   prompt = """
#   Utilze apenas os documentos para responder as perguntas, caso contrário, retorne a seguinte resposta: "Não conseguimos responder sua pergunta, por favor, entre em contato com a secretaria.".
#   {context}
#
#   Pergunta: {query}
#   """
#
#   qa_prompt = ChatPromptTemplate.from_messages([("human", prompt)])
#
#   chain = load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=qa_prompt)
#
#   query = pergunta
#
#   docs = retriever.invoke(query)
#
#   resposnse = chain.invoke(
#       {"input_documents": docs, "query": query}
#   )
#   return resposnse['output_text']
#
# resposta_duvida("tempo de duração do curso")

# ---------------------------------------------------------


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate

from src.llm import llm
from src.rag import retriever


def resposta_duvida(pergunta):
  prompt = """
  Utilze apenas os documentos para responder as perguntas, caso contrário, retorne a seguinte resposta: "Não conseguimos responder sua pergunta, por favor, entre em contato com a secretaria.".
  {context}

  Pergunta: {query}
  """

  qa_prompt = ChatPromptTemplate.from_messages([("human", prompt)])

  # chain = load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=qa_prompt)
  output_parser = StrOutputParser()
  #chain = create_stuff_documents_chain(llm, qa_prompt)
  query = pergunta

  #retriver = texts()
  docs = retriever().invoke(query)

  chain = qa_prompt | llm | output_parser

  resposnse = chain.invoke(
      {"input_documents": docs, "query": query}
  )
  return resposnse

print(resposta_duvida("tempo de duração do curso?"))