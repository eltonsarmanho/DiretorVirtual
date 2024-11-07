import time

import streamlit as st
from ChatAssistant import  ChatAssistant

from streamlit.web import cli as stcli
from streamlit import runtime
import sys
import os
import json

def main():
    st.set_page_config(page_title="Lendo Arquivos da UFPA", page_icon=":robot:")

    st.header("Diretor Virtual")

    input = st.text_input('Faça sua pergunta!')
    submit = st.button("Generate")

    if submit:
        if input != '':
            response_user = assistant.get_response(input)
            st.write(response_user)
        else:
            alert = st.warning('Por favor, Realize uma pergunta')
            time.sleep(3)
            alert.empty()

if __name__ == '__main__':
    assistant = ChatAssistant()

    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())