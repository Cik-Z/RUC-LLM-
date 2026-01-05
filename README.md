# 🎓 智能校园搜索引擎 (Intelligent Campus Search Engine)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

一个基于 **RAG (检索增强生成)** 技术与 **DeepSeek 大语言模型** 的垂直领域搜索引擎。该项目旨在解决传统校园搜索语义理解能力弱、信息获取效率低的问题，实现了从“搜索链接”到“智能问答”的跨越。

## ✨ 核心特性

* **🕷️ 高并发数据采集**：基于多线程的爬虫系统，支持断点续传与增量更新，自动处理非结构化网页清洗。
* **🔍 混合检索 (Hybrid Search)**：结合 **BM25** (稀疏检索) 与 **BGE-Small** (稠密向量检索)，利用 **RRF** 算法进行多路召回融合。
* **🧠 LLM 语义重排**：引入 **DeepSeek** 大模型对检索候选集进行二次精排 (Rerank)，显著提升 Top-K 准确率。
* **💬 RAG 智能问答**：基于检索上下文生成精准回答，拒绝幻觉，支持流式输出打字机效果。
* **⚡ 现代化交互**：基于 FastAPI 的异步后端与类似 New Bing 的前端界面，支持并行流式响应。

## 🏗️ 系统架构

![System Architecture](image_d2306d.jpg)

## 📂 项目结构

```text
├── corpus_dir/             # 存放爬取的语料数据 (生成的 corpus.jsonl)
├── dense_index/            # 存放 FAISS 向量索引文件
├── bm_index/               # 存放 Pyserini 倒排索引文件
├── index.html              # 前端交互界面
├── main.py                 # FastAPI 后端启动入口
├── data.py                 # 爬虫与数据预处理脚本
├── build_dense_index.py    # 向量索引构建脚本
├── hybrid_search.py        # 混合检索核心逻辑 (RRF)
├── llm_rerank.py           # LLM 重排模块
├── rag_qa.py               # RAG 问答模块
├── bm_search.py            # BM25 检索模块
└── dense_search.py         # 向量检索模块
