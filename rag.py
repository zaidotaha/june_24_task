import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize_qdrant():
    loader = PyPDFLoader("chatdev.pdf")
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()

    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        path="/tmp/local_qdrant",
        collection_name="my_documents",
    )

    return qdrant

qdrant = initialize_qdrant()

def process_user_query(user_query):

    model = ChatOpenAI(model="gpt-3.5-turbo")
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""
        Using the context, which are snippets that match a user query \
        about a Computer Science scientific paper, answer the user query. \
        **IMPORTANT**: Content is taken from the paper, do not mention you \
        are only given a specific context and not the entire paper.
        Context: {context}
        User query: {query}
    """
    )

    chain = prompt | model | parser

    found_docs = qdrant.similarity_search(user_query, k=6)
    docs = "\n".join([doc.page_content for doc in found_docs])
    output = chain.invoke({"context": docs, "query": user_query})
    return output

