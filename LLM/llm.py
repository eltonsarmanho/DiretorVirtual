from dotenv import load_dotenv
from langchain_community.chat_models import ChatMaritalk
import os
#from langchain.globals import set_verbose, get_verbose

load_dotenv()
# Set the verbose setting
#set_verbose(True)

# Get the current verbose setting
#current_verbose = get_verbose()

llm = ChatMaritalk(
        model="sabia-3",
        api_key= os.getenv("CHAVE_API"),
        temperature=0.7,
        max_tokens=100
    )
