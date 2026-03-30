import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import sys

# ========== 配置 ==========
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

EMBED_MODEL = "BAAI/bge-large-zh"
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')  # 替换为你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# ========== 1. 加载数据 ==========
def load_documents():
    docs = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    return docs

# ========== 2. 切分 ==========
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# ========== 3. 创建向量库 ==========
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return db

# ========== 4. 加载向量库 ==========
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# ========== 5. 风格 Prompt ==========
STYLE_PROMPT = """
你现在要完全模仿某位作者的思维方式和表达风格：

【思维特点】
- 先定义问题本质
- 分层拆解
- 强逻辑推导
- 经常指出常见误区

【语言特点】
- 有批判性但不过激
- 喜欢用类比
- 结尾给出洞察

要求：
- 不要泛泛而谈
- 必须结合提供的参考内容
- 输出要有“这个人写的味道”
"""

# ========== 6. 问答 ==========
def ask_question(db, question):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
{STYLE_PROMPT}

【参考内容】
{context}

【问题】
{question}

请直接、简洁地回答：
"""

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # 流式请求
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stream=True  # 启用流式输出
        )
        
        # 逐字输出
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                # 实时输出到屏幕
                print(content, end="", flush=True)
        
        print()  # 换行
        return full_response
        
    except Exception as e:
        return f"调用 API 时出错：{str(e)}"

# ========== 7. 主程序 ==========
def main():
    try:
        if not os.path.exists(DB_PATH):
            print("📚 构建知识库...")
            docs = load_documents()
            chunks = split_documents(docs)
            db = create_vector_db(chunks)
        else:
            print("📦 加载知识库...")
            db = load_vector_db()

        print("✅ 准备完成")

        while True:
            question = input("\n❓ 输入问题（exit 退出）：")

            if question.lower() == "exit":
                break

            print("\n🤖 回答：\n")
            answer = ask_question(db, question)

    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()