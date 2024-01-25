from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tiktoken
import streamlit as st

load_dotenv()


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_answer(query):
    index_path = "faiss_index"
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings()
        loaded_vectorstore = FAISS.load_local(index_path, embeddings)
    else:
        loader = PyPDFLoader("/Users/shivammitter/Desktop/AI/Gen_AI/embeddings/vectorestore-in-memory/48lawsofpower.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=tiktoken_len, separators=["\n\n", "\n"]
        )
        doc_f = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        loaded_vectorstore = FAISS.from_documents(doc_f, embeddings)
        loaded_vectorstore.save_local(index_path)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=loaded_vectorstore.as_retriever())

    result = qa.invoke({"query": query})
    return result.get("result", "No answer available.")


def main():
    st.set_page_config(
        page_title="Ask me anything",
        page_icon=":balance_scale:", 
        layout="wide",
    )

    st.title("The 48 Laws Of Power")
    st.image("/Users/shivammitter/Desktop/logo.png", width=200, caption="Book pic")

    query = st.text_input("Enter your law-related query:")

    if st.button("Submit"):
        result = get_answer(query)

        st.subheader("Result:")
        with st.container():
            st.success(result)


if __name__ == "__main__":
    main()