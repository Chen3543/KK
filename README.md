# 天涯 kk 神贴 RAG 知识库

基于 LangChain + ChromaDB，对天涯神贴楼主 kk 的帖子进行学习和分析，生成 RAG（检索增强生成）知识库问答系统。

## 项目简介

本项目爬取并整理了天涯神贴楼主 kk 的经典帖子，使用 embedding 模型向量化存储，通过大模型实现智能问答。

**核心功能：**
- 构建向量知识库
- 支持语义检索和 AI 问答
- 流式输出回答

## 快速开始

### 1. 安装依赖

```bash
#安装 LangChain 核心组件和本地模型依赖
pip install langchain chromadb pypdf sentence-transformers ollama
#安装云端 API 依赖
pip install openai chromadb tiktoken pypdf langchain
#推荐方式，从 requirements.txt 统一安装所有依赖
pip install -r requirements.txt
```

### 2. 配置大模型

**方案一：本地 Ollama（免费）**
```bash
# 安装 Ollama: https://ollama.ai
ollama pull qwen3.5:latest
```

**方案二：云端 API（推荐 DeepSeek）**
- 获取 API Key：https://platform.deepseek.com
- 修改 `main.py` 第 16 行：
```python
DEEPSEEK_API_KEY = "你的 API_KEY"
```

### 3. 运行

```bash
python main.py
```

首次运行会自动构建知识库，之后直接加载。

## 使用方法

输入问题，系统会基于 kk 的帖子内容回答：

```
❓ 输入问题（exit 退出）：kk 对...怎么看

🤖 回答：

根据 kk 在 2010 年的帖子，他认为...
```

## 项目结构

```
KK/
├── main.py              # 主程序
├── requirements.txt     # 依赖
├── data/               # kk 的帖子文件
│   ├── book1.pdf
│   ├── book2.pdf
│   └── qa.pdf
├── chroma_db/          # 向量数据库（自动生成）
└── README.md
```

## 技术栈

- **LangChain** - 大模型应用框架
- **ChromaDB** - 向量数据库
- **BGE-Large-Zh** - 中文 Embedding 模型
- **Ollama / DeepSeek** - 大语言模型

## 配置说明

**main.py 关键配置：**
```python
DATA_PATH = "./data"           # 帖子文件目录
DB_PATH = "./chroma_db"        # 向量库存储
EMBED_MODEL = "BAAI/bge-large-zh"  # Embedding 模型
DEEPSEEK_API_KEY = "sk-xxx"
```

## 关于天涯 kk 神贴

楼主 kk 是天涯社区传奇人物，其帖子被誉为中国房地产启蒙神贴，影响了无数人的房产观和财富观。

本项目旨在通过 RAG 技术，让 kk 的思想以 AI 问答的形式继续传承。

---
