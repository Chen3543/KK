import streamlit as st
import os
from src.vector_db import initialize_knowledge_base
from src.rag_pipeline import ask_question
from langchain_classic.memory import ConversationBufferWindowMemory
from src.chat_history_manager import save_chat_history, load_chat_history, get_all_chat_histories_meta, generate_new_chat_id, delete_chat_history

# --- Streamlit 页面配置 ---
st.set_page_config(page_title="天涯 kk 神贴 RAG 知识库", page_icon="💡", initial_sidebar_state="expanded", menu_items={
    'Get help': None,
    'Report a bug': None,
    'About': None
})

# --- 隐藏 Streamlit 默认 UI 元素 ---
st.markdown("""
<style>
/* 隐藏 Streamlit 默认的 Header 和 Footer */
header {
    display: none !important;
}
footer {
    display: none !important;
}

/* 隐藏右上角的"部署"按钮和三点菜单 (备用，如果 header 隐藏不彻底) */
button[data-testid="stDeployButton"] {
    display: none !important;
}
button[data-testid="stMenuButton"] {
    display: none !important;
}
div[data-testid="stHamburgerMenu"] {
    display: none !important;
}

/* ============================================ */
/* 完全重写 chat_input 样式 - 使用 Shadow DOM 穿透 */
/* ============================================ */

/* 最外层容器 */
div[data-testid="stChatInput"] {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

/* 第一层内部容器 */
div[data-testid="stChatInput"] > div {
    border: 2px solid #cccccc !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    outline: none !important;
    transition: all 0.3s ease !important;
    background-color: #ffffff !important;
    padding: 0 !important;
}

/* 第二层 - 文本输入区域 */
div[data-testid="stChatInput"] > div > div {
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}

/* 第三层 - 实际输入框容器 */
div[data-testid="stChatInput"] > div > div > div {
    background-color: #ffffff !important;
    border: none !important;
}

/* 所有子元素统一白色背景 */
div[data-testid="stChatInput"] *,
div[data-testid="stChatInput"] * * {
    background-color: #ffffff !important;
}

/* textarea 输入框 */
div[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
    background-color: transparent !important;
}

/* 强制覆盖聚焦时的红色边框 */
div[data-testid="stChatInput"]:focus-within > div,
div[data-testid="stChatInput"]:focus > div {
    border: 2px solid #4CAF50 !important;
    box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.2) !important;
    background-color: #ffffff !important;
}

/* 大模型思考时 - disabled 状态 */
div[data-testid="stChatInput"][aria-disabled="true"] > div,
div[data-testid="stChatInput"]:has(textarea:disabled) > div {
    border: 2px solid #4CAF50 !important;
    background-color: #ffffff !important;
    cursor: not-allowed !important;
    animation: breathing-fast 0.5s ease-in-out infinite !important;
}

/* 快速呼吸闪烁动画 */
@keyframes breathing-fast {
    0%, 100% {
        border-color: rgba(76, 175, 80, 0.2) !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1) !important;
    }
    50% {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 10px 5px rgba(76, 175, 80, 0.6) !important;
    }
}

/* 发送按钮 */
div[data-testid="stChatInput"] button {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: transparent !important;
}
</style>

<script>
// 使用 JavaScript 强制应用样式和闪烁动画
(function() {
    let animationInterval = null;
    let isAnimating = false;
    
    // 等待 DOM 加载完成
    function initChatInputAnimation() {
        const chatInputs = document.querySelectorAll('div[data-testid="stChatInput"]');
        
        chatInputs.forEach(chatInput => {
            // 监听属性变化以检测 disabled 状态
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'aria-disabled') {
                        const isDisabled = chatInput.getAttribute('aria-disabled') === 'true';
                        
                        if (isDisabled && !isAnimating) {
                            // 开始闪烁动画
                            startBlinking(chatInput);
                        } else if (!isDisabled && isAnimating) {
                            // 停止闪烁动画
                            stopBlinking(chatInput);
                        }
                    }
                });
            });
            
            // 开始观察
            observer.observe(chatInput, {
                attributes: true,
                attributeFilter: ['aria-disabled']
            });
            
            // 强制设置所有子元素背景色
            const allChildren = chatInput.querySelectorAll('*');
            allChildren.forEach(child => {
                child.style.backgroundColor = '#ffffff';
            });
        });
    }
    
    // 闪烁动画函数
    function startBlinking(chatInput) {
        if (isAnimating) return;
        isAnimating = true;
        
        const innerDiv = chatInput.querySelector(':scope > div');
        if (!innerDiv) return;
        
        let bright = true;
        
        // 清除之前的动画
        if (animationInterval) {
            clearInterval(animationInterval);
        }
        
        // 每 0.5 秒切换一次
        animationInterval = setInterval(() => {
            if (bright) {
                // 亮状态
                innerDiv.style.borderColor = '#4CAF50';
                innerDiv.style.boxShadow = '0 0 10px 5px rgba(76, 175, 80, 0.6)';
            } else {
                // 暗状态
                innerDiv.style.borderColor = 'rgba(76, 175, 80, 0.2)';
                innerDiv.style.boxShadow = '0 0 0 2px rgba(76, 175, 80, 0.1)';
            }
            bright = !bright;
        }, 500);
    }
    
    // 停止闪烁动画
    function stopBlinking(chatInput) {
        isAnimating = false;
        
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        
        const innerDiv = chatInput.querySelector(':scope > div');
        if (innerDiv) {
            // 恢复到正常聚焦状态
            innerDiv.style.borderColor = '#4CAF50';
            innerDiv.style.boxShadow = '0 0 0 4px rgba(76, 175, 80, 0.2)';
        }
    }
    
    // 页面加载后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initChatInputAnimation);
    } else {
        initChatInputAnimation();
    }
    
    // 定期检查（处理 Streamlit 动态渲染）
    setInterval(initChatInputAnimation, 2000);
})();
</script>
""", unsafe_allow_html=True)

# --- 初始化知识库和对话内存 ---
if "db" not in st.session_state:
    with st.spinner("📚 正在加载/构建知识库..."):
        st.session_state.db = initialize_knowledge_base()
    st.success("✅ 知识库准备完成！")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

# --- 历史对话管理 --- 
if "chat_histories_meta" not in st.session_state:
    st.session_state.chat_histories_meta = get_all_chat_histories_meta()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  # 初始没有选中任何对话

if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "新对话"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 侧边栏：历史会话管理 ---
with st.sidebar:
    st.header("💬 历史对话")

    # "新建对话" 按钮
    if st.button("➕ 新建对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.current_chat_id = None
        st.session_state.current_chat_title = "新对话"
        st.session_state.chat_histories_meta = get_all_chat_histories_meta() # 刷新历史列表
        st.rerun()

    st.markdown("--- ")

    # 历史对话列表
    if st.session_state.chat_histories_meta:
        for chat_meta in st.session_state.chat_histories_meta:
            chat_id = chat_meta["id"]
            chat_title = chat_meta["title"]
            # 突出显示当前选中的对话
            is_selected = (st.session_state.current_chat_id == chat_id)
            button_style = "primary" if is_selected else "secondary"

            col1, col2 = st.columns([8, 2])
            with col1:
                if st.button(chat_title, key=f"chat_button_{chat_id}", use_container_width=True, type=button_style):
                    loaded_history = load_chat_history(chat_id)
                    if loaded_history:
                        st.session_state.messages = loaded_history["messages"]
                        st.session_state.current_chat_id = loaded_history["id"]
                        st.session_state.current_chat_title = loaded_history["title"]

                        # 重新加载 LangChain memory
                        st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                st.session_state.memory.chat_memory.add_user_message(msg["content"])
                            elif msg["role"] == "assistant":
                                st.session_state.memory.chat_memory.add_ai_message(msg["content"])
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_button_{chat_id}"):
                    delete_chat_history(chat_id)
                    st.session_state.chat_histories_meta = get_all_chat_histories_meta() # 刷新历史列表
                    if st.session_state.current_chat_id == chat_id: # 如果删除的是当前对话，则切换到新对话
                        st.session_state.messages = []
                        st.session_state.memory.clear()
                        st.session_state.current_chat_id = None
                        st.session_state.current_chat_title = "新对话"
                    st.rerun()
    else:
        st.sidebar.info("暂无历史对话。")


# --- 主内容区域 ---
st.title("💡 天涯 kk 神贴 RAG 知识库")
st.caption("基于 LangChain + ChromaDB，对天涯神贴楼主 kk 的帖子进行学习和分析，生成 RAG（检索增强生成）知识库问答系统。")
st.write(f"## 当前对话：{st.session_state.current_chat_title}")

# --- 显示历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 用户输入 ---
if prompt := st.chat_input("向 kk 提问..."):
    # 如果是新对话，生成 ID 并设置标题
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = generate_new_chat_id()
        st.session_state.current_chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt # 取前30字作标题

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 调用问答函数
        try:
            full_response = ask_question(st.session_state.db, prompt, st.session_state.memory)
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_message = f"❌ 回答生成出错：{str(e)}"
            st.error(error_message)
            full_response = error_message # Ensure full_response has content for memory

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 保存当前对话历史
    save_chat_history(
        st.session_state.current_chat_id,
        st.session_state.current_chat_title,
        st.session_state.messages
    )
    st.session_state.chat_histories_meta = get_all_chat_histories_meta() # 刷新历史列表
    st.rerun() # 重新运行以更新侧边栏和主标题
