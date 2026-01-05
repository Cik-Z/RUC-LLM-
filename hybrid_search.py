# hybrid_search.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import re
from bm_search import bm25_search
from dense_search import dense_search

def hybrid_search(query: str, top_k: int = 10, k: int = 60):
    """
    使用 RRF (倒雷融合) 进行混合检索。
    公式: Score = 1 / (k + rank)
    
    :param k: RRF 常数，通常设为 60。
    """
    
    # 1. 并行获取结果 (通常取比最终 top_k 更多的候选集，比如 50)
    candidate_k = top_k * 5 
    bm25_hits = bm25_search(query, k=candidate_k)
    dense_hits = dense_search(query, top_k=candidate_k)

    # 用于存储融合分数
    # 格式: {docid: {"score": 0.0, "content": "...", "url": "...", "from": set()}}
    fusion_dict = {}

    # ===========================
    # 2. 处理 BM25 结果 (基于排名)
    # ===========================
    for rank, hit in enumerate(bm25_hits):
        docid = hit["docid"]
        
        # 初始化
        if docid not in fusion_dict:
            fusion_dict[docid] = {
                "score": 0.0, 
                "content": hit.get("contents", ""), 
                "url": hit.get("url", ""), 
                "from": set()
            }
        
        # RRF 累加
        fusion_dict[docid]["score"] += 1.0 / (k + rank + 1)
        fusion_dict[docid]["from"].add("bm25")

    # ===========================
    # 3. 处理 Dense 结果 (关键：解决 Chunk ID 问题)
    # ===========================
    # 记录 Dense 这一侧已经处理过的 docid，防止同一文档的多个 chunk 重复加分
    # 策略：如果一篇文档多个 chunk 命中，我们只取排名最高的那一次（或者你也可以累加，但通常取最高即可）
    seen_dense_docs = set()

    for rank, hit in enumerate(dense_hits):
        raw_id = hit["docid"]
        # 🔥 关键修复：从 "doc123_chunk4" 还原为 "doc123"
        real_docid = raw_id.split("_chunk")[0] 

        if real_docid in seen_dense_docs:
            continue # 同一文档的后续 chunk 不再参与排名计算（避免长文档霸榜）
            
        seen_dense_docs.add(real_docid)

        # 初始化 (如果 BM25 没搜到这个)
        if real_docid not in fusion_dict:
            # 注意：这里需要你 dense_search 返回 content/url
            # 如果 dense_search 没返回，可能需要单独查，或者只用 BM25 的元数据
            fusion_dict[real_docid] = {
                "score": 0.0, 
                "content": hit.get("contents", "Dense结果暂无预览"), 
                "url": hit.get("url", ""), 
                "from": set()
            }

        # RRF 累加
        fusion_dict[real_docid]["score"] += 1.0 / (k + rank + 1)
        fusion_dict[real_docid]["from"].add("dense")

    # ===========================
    # 4. 排序与格式化
    # ===========================
    # 按 RRF 分数倒序
    sorted_docs = sorted(fusion_dict.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # ===========================
    # 🔥 新增：结果去重逻辑 (De-duplication)
    # ===========================
    final_results = []
    seen_identifiers = set()

    for docid, data in sorted_docs:
        url = data["url"]
        content = data["content"]
        
        # --- 策略 A: URL 归一化 (解决 index.htm 问题) ---
        # 1. 去除 http/https 前缀差异
        norm_url = url.replace("https://", "").replace("http://", "")
        # 2. 去除末尾的斜杠
        norm_url = norm_url.rstrip("/")
        # 3. 去除默认首页文件名 (index.html, index.htm, default.aspx 等)
        norm_url = re.sub(r'/index\.(html|htm|php|jsp)$', '', norm_url, flags=re.IGNORECASE)
        
        # --- 策略 B: 内容指纹 (解决 URL 不同但内容完全一样的问题) ---
        # 取前 50 个字符作为指纹（一般首页的前 50 字都是一样的标题）
        # 如果你想更严格，可以用 hashlib.md5(content.encode()).hexdigest()
        content_fingerprint = content[:50].strip()

        # 检查是否重复
        # 如果 URL 归一化后相同，或者内容指纹完全相同，就视为重复
        if norm_url in seen_identifiers:
            continue
        # if content_fingerprint in seen_identifiers: # 可选：如果 URL 不同但内容一样也想去重，开启这行
        #     continue

        # 记录已出现的特征
        seen_identifiers.add(norm_url)
        # seen_identifiers.add(content_fingerprint) 

        # 加入最终结果
        final_results.append({
            "docid": docid,
            "score": data["score"],
            "url": url,       # 还是返回原始 URL 给用户
            "contents": content,
            "from": list(data["from"])
        })

        if len(final_results) >= top_k:
            break

    return final_results

if __name__ == "__main__":
    q = "中国人民大学 高瓴人工智能学院 人工智能 专业介绍"
    
    # 注意：RRF 不需要 alpha 参数！
    results = hybrid_search(q, top_k=5)
    
    print(f"\n🚀 混合检索结果 (Top {len(results)}):")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['docid']} (Score: {r['score']:.4f}) Sources: {r['from']}")
        print(f"    URL: {r['url']}")
        print(f"    Preview: {r['contents'][:60].replace(chr(10), ' ')}...")
        print("-" * 60)