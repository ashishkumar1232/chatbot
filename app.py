from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.title("Helpful Chatbot")
input_text = st.text_input("Search the topic you want")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, concise, and friendly chatbot. Always respond clearly, avoid unnecessary
    details, and answer the user's question as directly as possible. If a step-by-step explanation is needed, provide it simply."""),
    ("user", "{question}")
])

ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
llm = Ollama(model="llama3", base_url=ollama_url)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    try:
        response = chain.invoke({'question': input_text})
        st.write(response)
    except Exception as e:
        st.error("❌ Connection error: Unable to reach Ollama. Ensure it's running and accessible.")
        st.exception(e)

    st.write(response)
