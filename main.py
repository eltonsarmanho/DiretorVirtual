import time

import streamlit as st
from resposta_langchain_rag import resposta


st.set_page_config(page_title="Lendo Arquivos da UFPA", page_icon=":robot:")

st.header("Diretor Virtual ap Ines de Assistente")

input = st.text_input('Faça sua pergunta!')
submit = st.button("Generate")

if submit:
    if input != '':
        response_user = resposta(input)
        st.write(response_user)
    else:
        alert = st.warning('Por favor, Realize uma pergunta')
        time.sleep(3)
        alert.empty()