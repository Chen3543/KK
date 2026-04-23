from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
import json

# Neo4j 驱动
driver = None

def get_neo4j_driver():
    global driver
    if driver is None:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            print("✅ Neo4j 数据库连接成功")
        except Exception as e:
            print(f"❌ Neo4j 数据库连接失败: {e}")
            driver = None
    return driver

def close_neo4j_driver():
    global driver
    if driver is not None:
        driver.close()
        driver = None
        print("✅ Neo4j 数据库连接关闭")

def initialize_knowledge_graph():
    driver = get_neo4j_driver()
    if driver:
        # 这里可以添加一些初始化图谱的逻辑，例如创建索引或清空旧数据
        print("💡 知识图谱初始化完成 (或加载成功)")
    return driver

def extract_and_load_knowledge(text_content, doc_source="unknown"):
    llm_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

    # 使用 LLM 从文本中提取实体和关系
    # 这是一个初步的 Prompt 示例，需要根据实际效果进行迭代优化
    extraction_prompt = f"""
    从以下文本中提取关键实体及其之间的关系。请以 JSON 格式输出，包含一个 "entities" 数组和 "relationships" 数组。
    "entities" 数组中的每个对象应包含 "id" (唯一标识符), "type" (实体类型，如 Person, Concept, Event), "name" (实体名称)。
    "relationships" 数组中的每个对象应包含 "source" (源实体 id), "target" (目标实体 id), "type" (关系类型，如 BELIEVES, INFLUENCES)。

    如果无法提取任何实体或关系，返回空 JSON 数组。

    文本内容:
    {text_content}

    输出示例:
    {{
        "entities": [
            {{"id": "e1", "type": "Person", "name": "kk"}},
            {{"id": "e2", "type": "Concept", "name": "房地产"}}
        ],
        "relationships": [
            {{"source": "e1", "target": "e2", "type": "BELIEVES_IN"}}
        ]
    }}
    """

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个能够从文本中提取结构化信息的助手。"},
                {"role": "user", "content": extraction_prompt}
            ],
            response_format={"type": "json_object"}
        )
        kg_data_str = response.choices[0].message.content
        kg_data = json.loads(kg_data_str)

        driver = get_neo4j_driver()
        if driver:
            with driver.session() as session:
                # 导入实体
                for entity in kg_data.get("entities", []):
                    session.run(
                        f"""
                        MERGE (n:{entity['type']} {{name: $name}})
                        SET n.id = $id, n.source_doc = $doc_source
                        RETURN n
                        """,
                        id=entity.get("id"), name=entity.get("name"), doc_source=doc_source
                    )

                # 导入关系
                for rel in kg_data.get("relationships", []):
                    session.run(
                        f"""
                        MATCH (s), (t)
                        WHERE s.id = $source_id AND t.id = $target_id
                        MERGE (s)-[r:{rel['type']}]->(t)
                        SET r.source_doc = $doc_source
                        RETURN r
                        """,
                        source_id=rel.get("source"), target_id=rel.get("target"), doc_source=doc_source
                    )
            print(f"✅ 成功从 {doc_source} 提取并导入知识到 Neo4j")
        else:
            print("❌ Neo4j 驱动不可用，无法导入知识。")

    except Exception as e:
        print(f"❌ 知识提取或导入失败: {e}")
        print(f"LLM 响应: {kg_data_str}")
