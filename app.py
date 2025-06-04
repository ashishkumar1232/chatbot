import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit UI
st.title("🧠 Helpful Chatbot")
input_text = st.text_input("🔍 Ask a question:")

# Verify Ollama server connection
ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

try:
    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
    if response.status_code == 200:
        st.success("✅ Ollama server is reachable.")
    else:
        st.warning("⚠️ Ollama server responded but not with 200 OK.")
except requests.exceptions.RequestException:
    st.error("❌ Ollama server is not reachable. Make sure it is running at: " + ollama_url)
    st.stop()

# Set up LLM and prompt
llm = Ollama(model="llama3", base_url=ollama_url)
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, concise, and friendly chatbot. Always respond clearly, avoid unnecessary details, 
    and answer the user's question as directly as possible. If a step-by-step explanation is needed, provide it simply."""),
    ("user", "{question}")
])

# Chain setup
chain = prompt | llm | output_parser

# Handle user input
if input_text:
    try:
        response = chain.invoke({'question': input_text})
        st.markdown("### 💬 Response:")
        st.write(response)
    except Exception as e:
        st.error("❌ Failed to get a response from the model.")
        st.exception(e)
