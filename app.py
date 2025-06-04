# chatbot_app.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("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", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit UI
st.title("Helpful Chatbot")
input_text = st.text_input("Search the topic you want")

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, concise, and friendly chatbot. Always respond clearly, avoid unnecessary
    details, and answer the user's question as
    directly as possible. If a step-by-step explanation is needed, provide it in a simple and logical way."""),
    ("user", "{question}")
])

# LLM setup
llm = Ollama(model="llama3")  # Make sure llama3 is available in your Ollama instance
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Handle input
if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)
