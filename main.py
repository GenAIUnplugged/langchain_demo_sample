from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import requests
import os
import tiktoken

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0,api_key=os.getenv("OPENAI_API_KEY"),max_tokens=150,model="gpt-4-turbo" )

url = "https://wikipedia.org/wiki/India"
def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = BeautifulSoup(response.text, 'html.parser')
        paragraphs = data.find_all('p')
        text = [p.get_text() for p in paragraphs if p.get_text() != '']
        return ' '.join(text)
    else:
        raise Exception(f"Failed to fetch the page. Status code: {response.status_code}")

web_data = get_html(url)
doc = Document(page_content=web_data, metadata={"source": url})
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents([doc])
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# docs = vectorstore.similarity_search("What is capital of India?")
# print("Top 3 similar documents:",docs)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
query = "What are the states in India?"
result = qa({"query": query})   
print("Answer:", result['result'])





# def count_tokens(text: str, model: str = "gpt-4-turbo") -> int:
#     encoding = tiktoken.encoding_for_model(model)
#     return len(encoding.encode(text))

# total_tokens = sum(count_tokens(doc.page_content, model="gpt-4-turbo") for doc in docs)
# print("Number of tokens:", total_tokens)
