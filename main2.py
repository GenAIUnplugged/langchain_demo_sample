from openai import OpenAI
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Chatbot")
st.subheader("Ask anything")
st.divider()

st.sidebar.title("About")
st.sidebar.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
web_url = st.sidebar.text_input("Enter a URL", placeholder="https://example.com")
model = st.sidebar.selectbox("Select a model", ["gpt-4-turbo", "gpt-3.5-turbo"])

# Initialize the ChatOpenAI client
client = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=model,  # Use the selected model (e.g., gpt-4-turbo)
    temperature=0  # Adjust temperature as needed
)

def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = BeautifulSoup(response.text, 'html.parser')
        paragraphs = data.find_all('p')
        text = [p.get_text() for p in paragraphs if p.get_text() != '']
        return ' '.join(text)
    else:
        raise Exception(f"Failed to fetch the page. Status code: {response.status_code}")

@st.cache_resource
def process_url(url):
    web_data = get_html(url)
    doc = Document(page_content=web_data, metadata={"source": url})
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = text_splitter.split_documents([doc])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

if not web_url:
    st.warning("Please enter a valid URL.")
    st.stop()

if web_url:
    try:
        with st.spinner("Processing the URL..."):
            # Fetch and process the web page content
            vectorstore = process_url(web_url)
            
            # Create a retriever and QA chain
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa = RetrievalQA.from_chain_type(
                llm=client,  # Use the ChatOpenAI client
                chain_type="stuff",
                retriever=retriever
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
        qa = None  # Ensure qa is set to None if an error occurs

if 'qa' not in locals() or qa is None:
    st.error("The QA system is not initialized. Please check the input URL or file.")
else:
    # Handle user query
    query = st.text_input("Ask a question")
    if query:
        try:
            with st.spinner("Processing your query..."):
                result = qa({"query": query})
                st.write("Answer:", result.get('result', 'No result found'))
        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
