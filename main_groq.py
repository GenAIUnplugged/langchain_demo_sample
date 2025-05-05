import os
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("Chatbot")
st.subheader("Ask anything")
st.divider()

st.sidebar.title("About")
st.sidebar.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
web_url = st.sidebar.text_input("Enter a URL", placeholder="https://example.com")
model = st.sidebar.selectbox("Select a model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])

# Initialize the ChatGroq client
client = ChatGroq(
    model=model,
    temperature=0
)

# Function to fetch and parse HTML content
def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = BeautifulSoup(response.text, 'html.parser')
        paragraphs = data.find_all('p')
        text = [p.get_text() for p in paragraphs if p.get_text()]
        return ' '.join(text)
    else:
        raise Exception(f"Failed to fetch the page. Status code: {response.status_code}")

# Function to process the URL and create a vector store
@st.cache_resource
def process_url(url):
    web_data = get_html(url)
    doc = Document(page_content=web_data, metadata={"source": url})
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = text_splitter.split_documents([doc])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Custom prompt template
QA_PROMPT = PromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If the answer is not contained within the context, respond with "I don't know."

{context}

"Question": {question}
Helpful Answer:
""")

# Function to retrieve relevant documents using a similarity threshold
def get_relevant_docs(query, vectorstore, threshold=0.7):
    # Perform similarity search with relevance scores
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    # Filter documents based on the threshold
    filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
    return filtered_docs

# Main logic
if not web_url:
    st.warning("Please enter a valid URL.")
    st.stop()

try:
    with st.spinner("Processing the URL..."):
        # Fetch and process the web page content
        vectorstore = process_url(web_url)
except Exception as e:
    st.error(f"An error occurred: {e}")
    vectorstore = None

if vectorstore is None:
    st.error("The vector store is not initialized. Please check the input URL or file.")
else:
    # Handle user query
    query = st.text_input("Ask a question")
    if query:
        try:
            with st.spinner("Processing your query..."):
                # Retrieve relevant documents
                relevant_docs = get_relevant_docs(query, vectorstore, threshold=0.7)
                if not relevant_docs:
                    st.write("Answer: I don't know.")
                else:
                    # Create a retriever from the filtered documents
                    retriever = FAISS.from_documents(relevant_docs, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 3})

                    # Create a RetrievalQA chain with the custom prompt
                    qa = RetrievalQA.from_chain_type(
                        llm=client,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": QA_PROMPT}
                    )

                    # Create a Streamlit empty container for streaming the result
                    answer_placeholder = st.empty()
                    # Simulate streaming the answer
                    result = qa({"query": query})

                    # Stream the result
                    answer_placeholder.write("Answer: " + result.get('result', 'No result found'))

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
