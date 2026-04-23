import json
import os
import uuid
from datetime import datetime

# 存储历史对话的目录
CHAT_HISTORY_DIR = "chat_histories"

def _get_history_file_path(chat_id: str) -> str:
    """根据 chat_id 获取历史文件路径"""
    return os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")

def save_chat_history(chat_id: str, title: str, messages: list):
    """保存或更新一个对话历史"""
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)

    history_file_path = _get_history_file_path(chat_id)
    history_data = {
        "id": chat_id,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    with open(history_file_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)
    # print(f"✅ 对话 '{title}' (ID: {chat_id}) 已保存/更新。")

def load_chat_history(chat_id: str) -> dict | None:
    """加载指定 chat_id 的对话历史"""
    history_file_path = _get_history_file_path(chat_id)
    if os.path.exists(history_file_path):
        with open(history_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_all_chat_histories_meta() -> list[dict]:
    """获取所有对话历史的元数据 (id, title, timestamp) 列表"""
    histories_meta = []
    if not os.path.exists(CHAT_HISTORY_DIR):
        return histories_meta

    for filename in os.listdir(CHAT_HISTORY_DIR):
        if filename.endswith(".json"):
            chat_id = filename.split(".")[0]
            try:
                history_data = load_chat_history(chat_id)
                if history_data:
                    histories_meta.append({
                        "id": history_data["id"],
                        "title": history_data["title"],
                        "timestamp": history_data["timestamp"]
                    })
            except json.JSONDecodeError:
                print(f"⚠️ 无法解析历史文件: {filename}")
    # 按时间戳降序排序，最新的在前面
    histories_meta.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return histories_meta

def generate_new_chat_id() -> str:
    """生成一个新的唯一的对话 ID"""
    return str(uuid.uuid4())

def delete_chat_history(chat_id: str):
    """删除指定的对话历史"""
    history_file_path = _get_history_file_path(chat_id)
    if os.path.exists(history_file_path):
        os.remove(history_file_path)
        # print(f"🗑️ 对话 (ID: {chat_id}) 已删除。")
    else:
        print(f"⚠️ 对话 (ID: {chat_id}) 不存在，无法删除。")
