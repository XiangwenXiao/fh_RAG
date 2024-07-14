import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader,DirectoryLoader
from langchain.vectorstores import FAISS
import os

OPENAI_API_KEY = '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'


llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
vectorstore = None
vector_store_path = r'./vector_store'
embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')


def load_txt(txt_path):
    return DirectoryLoader(txt_path,glob="**/*.txt").load()

st.title('TXT Chatbot')

with st.container():
    upload_file = st.file_uploader('upload file',type='txt')
    if upload_file is not None:
        path = r"D:\PycharmFile\test\save_files"
        with open(os.path.join(path,upload_file.name),"wb") as f:
            f.write((upload_file).getbuffer())
            st.write("haved upload")

        docs = load_txt(r'./save_files')

        text_splitter = CharacterTextSplitter(chunk_size=250,chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        docsearch = FAISS.from_documents(split_docs,embeddings)

        docsearch.save_local(vector_store_path)
        st.write('Done')

with st.container():
    question = st.text_input('question')
    docsearch = FAISS.load_local(vector_store_path,embeddings=embeddings)
    if vectorstore is not None and question is not None and question != "":
        qa = ConversationalRetrievalChain.from_llm(llm = llm,
                                                   retriever=docsearch,
                                                   memory=memory,
                                                   return_source_documents=True,
                                                   combine_docs_chain_kwargs={"prompt": question})
        answer = qa.run({"question": question})
        print(answer)
        st.write(answer)
