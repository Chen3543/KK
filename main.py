# main.py - 模块导入入口

from src.vector_db import initialize_knowledge_base
from src.rag_pipeline import ask_question
from langchain_classic.memory import ConversationBufferWindowMemory
