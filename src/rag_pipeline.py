import os
from openai import OpenAI
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, STYLE_PROMPT

# ========== 6. 问答 ==========
def ask_question(db, question, memory):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the current user's message content (without STYLE_PROMPT)
    current_user_message_content = f"""
【参考内容】
{context}

【问题】
{question}

请直接、简洁地回答：
"""

    # Prepare messages for the LLM, including system prompt and historical messages
    messages = [
        {"role": "system", "content": STYLE_PROMPT} # System prompt for style
    ]

    # Add historical messages from memory
    chat_history_list = memory.load_memory_variables({})["chat_history"]
    for msg in chat_history_list:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Add the current user's message
    messages.append({"role": "user", "content": current_user_message_content})

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # Stream request
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages, # Use the prepared messages list
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
                # print(content, end="", flush=True) # Streamlit doesn't use print for UI

        # Extract sources from retrieved documents
        sources = []
        for doc in docs:
            if 'source' in doc.metadata:
                source_path = os.path.basename(doc.metadata['source']) # Get only the filename
                if source_path not in sources: # Avoid duplicate sources
                    sources.append(source_path)

        # Format sources for display
        if sources:
            sources_text = "\n\n**参考来源：**\n" + "\n".join([f"- {s}" for s in sources])
            full_response += sources_text

        # Save current interaction to memory
        memory.save_context({"input": question}, {"output": full_response})

        return full_response
        
    except Exception as e:
        return f"调用 API 时出错：{str(e)}"
