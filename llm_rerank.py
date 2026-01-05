# LLM_rerank.py
"""
基于 Hybrid 初筛 + DeepSeek LLM 重排的检索模块。
修复版：解决了 ImportError: cannot import name 'get_lucene_searcher'
"""

import os
import json
from typing import List, Dict
from openai import OpenAI

# 导入模块
from hybrid_search import hybrid_search
from bm_search import get_searcher

# 配置 DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def _build_rerank_prompt(query: str, docs: List[Dict]) -> str:
    """构造给 LLM 的打分提示词"""
    lines = []
    lines.append("你是一个搜索引擎的相关性评估助手。")
    lines.append("请为以下文档打分（0-5分），0=无关，5=高度相关。")
    lines.append(f"用户查询：{query}\n")
    lines.append("候选文档列表：")

    for i, d in enumerate(docs, 1):
        # 截取前 300 字
        snippet = d["contents"].replace("\n", " ")[:300]
        lines.append(f"[DOC_{i}] docid={d['docid']}")
        lines.append(f"内容: {snippet}\n")

    lines.append("请只输出 JSON 数组，格式：")
    lines.append('[{"docid": "...", "score": 0-5}, ...]')
    return "\n".join(lines)

def _parse_llm_json(text: str) -> List[Dict]:
    """解析 LLM 返回的 JSON"""
    try:
        text = text.strip()
        # 清理 markdown 标记
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return []
    except:
        return []

def llm_rerank(query: str, top_k_candidate: int = 50, top_k_final: int = 10, alpha: float = 0.7) -> List[Dict]:
    """
    Hybrid Search -> LLM Rerank
    """
    # 1. 初筛 (Hybrid)
    hybrid_hits = hybrid_search(query, top_k=top_k_candidate, k=60)

    # 2. 🔥【核心修复】创建一次 Searcher，而不是在循环里创建
    searcher = get_searcher()
    
    docs = []
    for h in hybrid_hits:
        try:
            # 3. 查原文
            doc = searcher.doc(h["docid"])
            if doc:
                raw = json.loads(doc.raw())
                docs.append({
                    "docid": h["docid"],
                    "url": raw.get("url", ""),
                    "contents": raw.get("contents", ""),
                    "hybrid_score": h["score"]
                })
        except Exception as e:
            print(f"⚠️ 文档读取失败: {h['docid']} - {e}")
            continue

    if not docs:
        return []

    # 4. 调用 LLM 进行重排
    prompt = _build_rerank_prompt(query, docs)
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的搜索相关性打分器，只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        scored_list = _parse_llm_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Rerank 失败: {e}")
        scored_list = []

    # 5. 分数融合 (LLM Score + Hybrid Score)
    score_map = {item["docid"]: float(item["score"]) for item in scored_list if "docid" in item and "score" in item}

    reranked = []
    for d in docs:
        docid = d["docid"]
        llm_score = score_map.get(docid, 0.0)
        # 综合分：主要看 LLM，Hybrid 微调
        final_score = llm_score + 0.1 * d["hybrid_score"]
        
        reranked.append({
            "docid": docid,
            "url": d["url"],
            "contents": d["contents"],
            "final_score": final_score
        })

    # 6. 排序并返回 Top K
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked[:top_k_final]