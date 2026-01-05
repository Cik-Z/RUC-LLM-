# build_dense_index.py
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========= 路径按你的目录结构设置 =========
CORPUS_DIR = "/Users/cik-z/Desktop/智能信息检索导论/作业/final/corpus_dir"
OUTPUT_INDEX = "/Users/cik-z/Desktop/智能信息检索导论/作业/final/dense_index/dense.index"
OUTPUT_IDS = "/Users/cik-z/Desktop/智能信息检索导论/作业/final/dense_index/docids.json"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def chunk_text(text, size=300, overlap=50):
    """把文本切片成 chunk"""
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


def build_dense_index():
    print("加载 Embedding 模型：", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    emb_dim = model.get_sentence_embedding_dimension()
    print("Embedding 维度：", emb_dim)

    print("\n开始读取 corpus_dir 下的 JSONL 文件...\n")

    texts = []
    ids = []

    # ===== 遍历 corpus_dir 下所有 jsonl 文件 =====
    json_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".jsonl")]

    for fname in json_files:
        path = os.path.join(CORPUS_DIR, fname)
        print(f"读取文件：{path}")

        # 获取行数以显示进度条
        total_lines = sum(1 for _ in open(path, "r", encoding="utf-8"))

        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc=f"处理 {fname}"):
                obj = json.loads(line)
                docid = obj.get("id", "")
                content = obj.get("contents", "")

                if not content or not docid:
                    continue

                chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                for idx, ch in enumerate(chunks):
                    texts.append(ch)
                    ids.append(f"{docid}_chunk{idx}")

    print(f"\n📌 总 chunk 数量：{len(texts)}\n")

    # ===== 对所有 chunks 编码向量 =====
    print("开始编码向量（embedding）...\n")
    embeddings = []

    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding 进度"):
        batch = texts[i:i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True)
        embeddings.append(vecs)

    embeddings = np.vstack(embeddings).astype("float32")

    print("\n向量编码完成，开始构建 FAISS index...\n")

    # ===== 构建 FAISS IndexFlatIP =====
    index = faiss.IndexFlatIP(emb_dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX)

    with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print("\n🎉 完成！")
    print(f"向量索引保存在：{OUTPUT_INDEX}")
    print(f"chunk-ID 映射保存在：{OUTPUT_IDS}")


if __name__ == "__main__":
    build_dense_index()
