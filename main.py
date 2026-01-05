# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入功能模块
from llm_rerank import llm_rerank
from rag_qa import rag_answer
from hybrid_search import hybrid_search
from bm_search import get_searcher

app = FastAPI(title="智能校园搜索")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_llm: bool = True 

class QARequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {"error": "请确保 index.html 文件在当前目录下"}

def extract_title(content: str) -> str:
    if not content: return "无标题文档"
    title = content.split('\n')[0].strip()
    if len(title) > 40: title = title[:40] + "..."
    return title

# --- 搜索接口 ---
@app.post("/search")
def search_api(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    response_data = []
    try:
        if req.use_llm:
            print(f"🔍 [Search] DeepSeek Rerank | Query: {req.query}")
            results = llm_rerank(req.query, top_k_candidate=20, top_k_final=req.top_k, alpha=0.7)
            for r in results:
                content = r.get("contents", "")
                response_data.append({
                    "docid": r.get("docid"),
                    "url": r.get("url"),
                    "score": r.get("final_score"),
                    "title": extract_title(content),
                    "preview": content[:150].replace("\n", " ") + "..." 
                })
        else:
            print(f"🔍 [Search] Hybrid Only | Query: {req.query}")
            hybrid_results = hybrid_search(req.query, top_k=req.top_k, alpha=0.7)
            searcher = get_searcher()
            for h in hybrid_results:
                content = ""
                url = ""
                try:
                    lucene_doc = searcher.doc(h["docid"])
                    if lucene_doc:
                        raw = json.loads(lucene_doc.raw())
                        content = raw.get("contents", "")
                        url = raw.get("url", "")
                except: pass
                
                response_data.append({
                    "docid": h["docid"],
                    "url": url,
                    "score": h.get("score"),
                    "title": extract_title(content),
                    "preview": content[:150].replace("\n", " ") + "..."
                })

        return {"code": 200, "data": response_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 500, "error": str(e)}

# --- 🔥 问答接口 (RAG) ---
@app.post("/ask")
def ask_api(req: QARequest):
    print(f"🤖 [QA] Generating Answer | Query: {req.query}")
    try:
        # 调用 rag_qa.py 里的逻辑
        answer = rag_answer(query=req.query, top_k=5)
        return {"code": 200, "answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 500, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)