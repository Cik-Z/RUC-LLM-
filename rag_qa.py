# rag_qa.py
import os
import json
from openai import OpenAI
from hybrid_search import hybrid_search 
from bm_search import get_searcher # 必须复用这个正确的 searcher

# 配置 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def build_prompt(query: str, context_docs: list) -> str:
    """构建给大模型的提示词 - 优化版"""
    context_str = ""
    for i, doc in enumerate(context_docs, 1):
        # 稍微增加一点长度，350字
        content = doc.get("contents", "")[:350].replace("\n", " ")
        context_str += f"[参考文档{i}]: {content}\n\n"

    # 🔥 修改核心：让 AI 既能回答问题，也能总结关键词
    prompt = f"""
    你是一个智能校园助手。请参考下面的【参考资料】来处理用户的【输入】。

    任务要求：
    1. 如果【输入】是一个具体问题（如“学院在哪？”），请直接回答。
    2. 如果【输入】只是一个关键词（如“人工智能”），请根据参考资料生成一段简短的摘要或介绍。
    3. 这是一个检索增强系统，请**完全基于**【参考资料】回答。

    【参考资料】：
    {context_str}

    【输入】：{query}
    """
    return prompt

def rag_answer(query: str, top_k: int = 5) -> str:
    """
    RAG 流程
    """
    print(f"🤖 [RAG] 正在思考: {query}")
    
    # 1. 检索 (复用 hybrid_search)
    hits = hybrid_search(query, top_k=top_k)
    searcher = get_searcher()
    
    context_docs = []
    for h in hits:
        try:
            # ✅ 必须使用正确的 .doc().raw() 写法
            doc = searcher.doc(h["docid"])
            if doc:
                raw = json.loads(doc.raw())
                context_docs.append({
                    "contents": raw.get("contents", ""),
                    "url": raw.get("url", "")
                })
        except Exception as e:
            print(f"⚠️ 文档解析跳过: {e}")
            continue

    if not context_docs:
        return "抱歉，没有找到相关的校园资料，无法回答您的问题。"

    # 2. 构建 Prompt
    prompt = build_prompt(query, context_docs)
    
    # 🔥 调试打印：让你在后台看到到底发给了 AI 什么
    print("---------------- PROMPT ----------------")
    print(prompt[:500] + "...\n(提示词过长已截断)")
    print("----------------------------------------")

    # 3. 调用 DeepSeek
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个乐于助人的校园问答助手。回答要简洁，语气亲切。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3, 
            stream=False # 暂时不用流式，简单点
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"❌ LLM 调用出错: {e}")
        return "抱歉，AI 大脑暂时短路了，请检查 API Key 或网络。"

if __name__ == "__main__":
    # 本地测试
    print(rag_answer("人工智能"))