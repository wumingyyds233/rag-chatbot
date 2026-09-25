import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ========== 1. 页面配置 ==========
st.set_page_config(
    page_title="个人知识库问答",
    page_icon="📚",
    layout="wide"
)
st.title("📚 个人知识库问答")

# ========== 2. 缓存资源（只加载一次） ==========
@st.cache_resource(show_spinner="正在加载嵌入模型...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner="正在连接向量库...")
def load_vectorstore(persist_dir="./kb_chroma"):
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )

@st.cache_resource(show_spinner="正在加载大模型...")
def load_llm():
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

# ========== 3. 构建 RAG Chain ==========
def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个知识库助手。请根据以下上下文回答问题。
如果上下文中没有答案，请如实说"根据已有资料无法回答"。
回答时请在末尾标注信息来源。

上下文：
{context}"""),
        ("user", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ========== 4. 侧边栏：显示来源文档 ==========
with st.sidebar:
    st.header("📖 知识库信息")
    try:
        vs = load_vectorstore()
        st.success(f"已索引 {vs._collection.count()} 条文档块")
    except Exception as e:
        st.error(f"向量库加载失败：{e}")
        st.stop()

    st.divider()
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# ========== 5. 初始化会话历史 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 6. 渲染历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== 7. 处理用户输入 ==========
if question := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                vectorstore = load_vectorstore()
                llm = load_llm()
                chain = build_rag_chain(vectorstore, llm)
                answer = chain.invoke(question)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = f"出错了：{type(e).__name__}: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )