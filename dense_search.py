import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "dense_index/dense.index"
ID_MAPPING_PATH = "dense_index/docids.json"
CORPUS_PATH = "corpus_dir/corpus.jsonl"   # ← NEW：读取网页 url/contents
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# ========== 加载 corpus（一次加载） ==========
def load_corpus():
    corpus = {}
    if not os.path.exists(CORPUS_PATH):
        print("❌ 未找到 corpus.jsonl，请确认路径")
        return corpus
    
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["id"]] = {
                "url": obj.get("url", ""),
                "contents": obj.get("contents", "")
            }
    return corpus


def load_dense_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ID_MAPPING_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids


def load_model():
    return SentenceTransformer(MODEL_NAME)


def encode(model, text):
    emb = model.encode([text], normalize_embeddings=True)
    return emb.astype("float32")


def dense_search(query, top_k=5):
    model = load_model()
    index, ids = load_dense_index()
    corpus = load_corpus()

    q_emb = encode(model, query)
    dists, idxs = index.search(q_emb, top_k)

    results = []

    for dist, idx in zip(dists[0], idxs[0]):
        chunk_id = ids[idx]                       # 例如：doc123_chunk4
        docid = chunk_id.split("_chunk")[0]       # → doc123

        # 找原始网页信息
        page = corpus.get(docid, {})

        preview = page.get("contents", "")[:150].replace("\n", " ")
        url = page.get("url", "")

        results.append({
            "chunk_id": chunk_id,
            "docid": docid,
            "url": url,
            "preview": preview,
            "score": float(dist)
        })

    return results


if __name__ == "__main__":
    q = "中国人民大学 高瓴人工智能学院"
    hits = dense_search(q, top_k=5)

    print("\n=========== 向量检索结果（Chunk-based Dense Search） ===========")
    for r in hits:
        print("------------------------------------------------------------")
        print(f"[{r['chunk_id']}] score={r['score']}")
        print(f"URL: {r['url']}")
        print(f"Preview: {r['preview']} ...")
