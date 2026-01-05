# bm_search.py
import json
from pyserini.search.lucene import LuceneSearcher

INDEX_DIR = "bm_index"

def get_searcher():
    searcher = LuceneSearcher(INDEX_DIR)
    searcher.set_language('zh')
    searcher.set_bm25(k1=0.9, b=0.4)
    return searcher

"""
def bm25_search(query: str, k: int = 10):
    searcher = get_searcher()
    hits = searcher.search(query, k)

    results = []

    for hit in hits:
        # --- 🔥 修改开始 ---
        # 1. 先用 docid 从 searcher 里把完整的文档取出来
        doc = searcher.doc(hit.docid)
        
        # 2. 检查一下有没有取到（防止报错）
        if doc is None:
            continue
            
        # 3. 再调用 .raw() 方法获取 JSON 字符串
        raw_json = json.loads(doc.raw())
        # --- 🔥 修改结束 ---

        results.append({
            "docid": hit.docid,
            "score": hit.score,
            "url": raw_json.get("url", "（无URL字段）"),
            "contents": raw_json.get("contents", "")
        })

    return results
"""

def bm25_search(query: str, k: int = 10):
    print(f"DEBUG: 正在搜索关键词: {query}")
    searcher = get_searcher()
    
    # 关键点1：确认搜索是否真的找到了 id
    hits = searcher.search(query, k)
    print(f"DEBUG: 搜索结果数量 (hits): {len(hits)}")

    results = []

    for i, hit in enumerate(hits):
        # 1. 取文档对象
        doc = searcher.doc(hit.docid)
        
        # 关键点2：确认是否能根据 id 取回文档内容
        if doc is None:
            print(f"DEBUG: 第 {i+1} 条 (id={hit.docid}) -> 文档对象为 None (索引时可能未开启 storeContents)")
            continue
            
        try:
            # 2. 获取原始字符串
            raw_str = doc.raw()
            print(f"DEBUG: 第 {i+1} 条 -> 原始数据前50字: {raw_str[:50]}") # 看看是不是空的
            
            # 3. 解析 JSON
            raw_json = json.loads(raw_str)
            
            # 关键点3：确认 JSON 里的字段名对不对
            print(f"DEBUG: 第 {i+1} 条 -> JSON的所有键: {list(raw_json.keys())}")
            
            content = raw_json.get("contents", "")
            if not content:
                 # 尝试读取 text 字段，防备字段名不叫 contents
                 content = raw_json.get("text", "真的找不到内容")
            
            results.append({
                "docid": hit.docid,
                "score": hit.score,
                "url": raw_json.get("url", "无URL"),
                "contents": content
            })
            
        except Exception as e:
            print(f"DEBUG: 解析出错: {e}")

    print(f"DEBUG: 最终返回结果数: {len(results)}")
    return results

if __name__ == "__main__":
    q = "毛佳昕"
    results = bm25_search(q, k=10)

    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"[{i}] docid={r['docid']}, score={r['score']}")
        print("URL:", r["url"])

        preview = r["contents"].replace("\n", " ")[:150]
        print("Preview:", preview + "...")
        print("-" * 60)