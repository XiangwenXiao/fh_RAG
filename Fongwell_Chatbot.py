import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import (AIMessage,HumanMessage)
from langchain.text_splitter import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.vectorstores import FAISS
import os
from search import answer_format
import re
import pandas as pd
from utils import read_yaml

config = read_yaml("./config.yaml")

vocab = pd.read_excel(r"码表.xlsx")


st.set_page_config(page_title="Welcome to Fongwell🤖", layout="wide")
embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')


#侧边栏
st.session_state["params"] = {}
param = config.get("Parameter")
st.sidebar.markdown("### ⚙️ Parameter Settings")
#window_memory
window_memory = st.sidebar.number_input("**Window Memory**", 0, 10, 3)
st.session_state["params"]["window_memory"] = window_memory
#search_for_top_n
top_p = st.sidebar.number_input("**Search For topn**", 0, 10, 3)
st.session_state["params"]["top_p"] = top_p
#temperature
param_temp = param.get("temperature")
temperature = st.sidebar.slider(
    "**Temperature**",
    min_value=param_temp.get("min_value"),
    max_value=param_temp.get("max_value"),
    value=param_temp.get("default"),
    step=param_temp.get("stemp"),
    help=param_temp.get("tip"),
)
st.session_state["params"]["temperature"] = temperature
#相似性
similarity_score_threshold = st.sidebar.slider("**Similarity Score Threshold Memory**",min_value= 0.0, max_value=1.0, step=0.1,value=0.8)
st.session_state["params"]["similarity_score_threshold"] = similarity_score_threshold
#向量库
contents = os.listdir(r'.\vector_store')
subfolders = [f for f in contents]
vs = st.sidebar.selectbox('**Vector Strore**',subfolders)

#聊天框
#qa设置
#提示
from langchain_core.prompts import PromptTemplate
prompt_template = """
                    请注意：请谨慎评估query与提示的Context信息的相关性，
                        只根据本段输入文字信息的内容进行回答，如果query与提供的材料无关，
                        请回答"您好，给您匹配以下可能相关的数据字段"，另外也不要回答无关答案：
                        Context: {context}
                        Question: {question}
                        Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#记忆
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=window_memory,
                                        memory_key='chat_history',
                                        return_messages=True,
                                        output_key='answer')

if "messages" not in st.session_state:
    st.session_state["messages"] = []

vector_store_path = os.path.join('./vector_store',vs)
with st.container():
    st.header("Chat with FongWell😃")
    docsearch = FAISS.load_local(vector_store_path,embeddings=embeddings,allow_dangerous_deserialization=True)

    for message in st.session_state["messages"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    prompt = st.chat_input("Type something...")
    if prompt:
        st.session_state["messages"].append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        # qa
        chat = ConversationalRetrievalChain.from_llm(ChatOllama(model="llama3"),
                                                   docsearch.as_retriever(search_kwargs={"k": top_p,
                                                                                        }),
                                                   memory=memory,
                                                   return_source_documents=True,
                                                   combine_docs_chain_kwargs={"prompt": PROMPT}
                                                     )

        res = chat({"question": prompt})
        ai_message = res['answer']
        source = res['source_documents']

        text = str(source)
        # 内容项
        start_marker = "page_content="
        end_marker = ", metadata="

        content_pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker))
        content_matches = content_pattern.finditer(text)

        content_extracted_texts = []

        for match in content_matches:
            extracted_text = match.group(1)
            content_extracted_texts.append(extracted_text)

        # 来源公司项
        source_start_maker = "save_files\\"
        source_end_maker = ".txt'}"

        source_pattern = re.compile(re.escape(source_start_maker) + r"(.*?)" + re.escape(source_end_maker))
        source_matches = source_pattern.finditer(text)

        source_extracted_texts = []

        for match in source_matches:
            extracted_text = match.group(1)
            extracted_text = extracted_text[1:]
            source_extracted_texts.append(extracted_text)

        st.session_state["messages"].append(ai_message)
        with st.chat_message("assistant"):
            st.markdown(ai_message)

            for i in range(len(source)):

                source_doc = vocab.loc[vocab['缩写'] == source_extracted_texts[i], '公司名称'].values[0]
                word = f"**来源{i+1}:**" + "  \n"+content_extracted_texts[i][1:-1].replace("\\n", '  \n') + "  \n"+ f"**Source:{source_doc}**"
                st.markdown(word)

            #st.write(res)