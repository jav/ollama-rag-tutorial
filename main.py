# Loading required libraries
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


data_path = "./resources/2024-4-9020.pdf"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=30,
    length_function=len,
)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)

print("OpenAIEmbeddings")
embedding_func = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
print("Chroma.from_documents")
vectordb = Chroma.from_documents(documents, embedding=embedding_func)

template = """<s><<SYS>> Given the context - {context} <</SYS>>[INST]  Answer the following question - {question}[/INST]"""
pt = PromptTemplate(template=template, input_variables=["context", "question"])

print("Run RetrievalQA")
rag = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama2:latest"),
    retriever=vectordb.as_retriever(),
    memory=ConversationSummaryMemory(llm=Ollama(model="llama2:latest")),
    chain_type_kwargs={"prompt": pt, "verbose": True},
)


print(f'--== {rag.invoke("Query specific to the PDF that was tokenized?")} ==--')

print("the end")
