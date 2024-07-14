import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



user_input = st.text_input("请输入知识库名称：")
st.session_state["vectorstorename"] = user_input


file_path = "./save_files"
vector_store_path = './vector_store'

upload_files = st.file_uploader('upload file',type='txt',accept_multiple_files=True)
if upload_files is not None:
    for upload_file in upload_files:
        with open(os.path.join(file_path,upload_file.name),"wb") as f:
            f.write((upload_file).getbuffer())
    #st.write("文件已上传！")



create= st.button('基于以上数据生成向量数据库')
if create:
    loader = DirectoryLoader(file_path, glob="**/*.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')
    docsearch = FAISS.from_documents(split_docs, embeddings)
    docsearch.save_local(os.path.join(vector_store_path,user_input))
    st.write("向量数据库已生成！")

st.sidebar.selectbox("**Vector Store**", ["FAISS", "Milvus", "Chroma"])
st.sidebar.selectbox("**Embedding**", ["m3e-base", "text2vec", "ernie-base"])