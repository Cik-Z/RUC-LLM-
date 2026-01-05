import json
import re
import time
import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "temp_urls.json"
OUTPUT_FILE = "corpus.jsonl"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 并发数：70万数据建议保守点，避免被封IP导致前功尽弃
MAX_WORKERS = 50
# ===========================================

def normalize_url(url):
    """URL 归一化"""
    if not url: return ""
    u = url.strip()
    u = u.replace("https://", "").replace("http://", "")
    u = u.rstrip("/")
    u = re.sub(r'/index\.(html|htm|php|jsp|asp|aspx)$', '', u, flags=re.IGNORECASE)
    return u

def fetch_and_process(task):
    """
    单个任务处理
    """
    doc_id = task["id"]
    url = task["url"]
    
    target_url = url if url.startswith("http") else "http://" + url
    content = ""
    
    try:
        # timeout 设置为 10秒，防止卡死
        response = requests.get(target_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        
        if response.encoding == 'ISO-8859-1':
             response.encoding = response.apparent_encoding
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # 移除干扰元素
            for script in soup(["script", "style", "nav", "footer", "iframe", "noscript", "svg"]):
                script.extract()
            content = soup.get_text(separator=" ", strip=True)
            
    except Exception:
        # 网络错误在大量爬取中很常见，记录为空内容即可，不要中断程序
        pass

    return {
        "id": doc_id,
        "url": url,
        "contents": content
    }

def get_finished_urls():
    """
    检查输出文件，获取已经爬取过的 URL 集合 (用于断点续爬)
    """
    finished = set()
    max_doc_id = 0
    
    if not os.path.exists(OUTPUT_FILE):
        return finished, 0

    print(f"🔄 检测到 {OUTPUT_FILE} 已存在，正在扫描断点...")
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    # 记录归一化 URL
                    u = normalize_url(data.get("url", ""))
                    if u: finished.add(u)
                    
                    # 解析 docID 数字，为了让 ID 继续往下排
                    # 假设 ID 格式为 doc123
                    cid = data.get("id", "doc0").replace("doc", "")
                    if cid.isdigit():
                        cid_num = int(cid)
                        if cid_num > max_doc_id:
                            max_doc_id = cid_num
                except:
                    continue
    except Exception as e:
        print(f"读取断点文件出错: {e}")
        
    print(f"✅ 已完成: {len(finished)} 条，最大 ID: doc{max_doc_id}")
    return finished, max_doc_id

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}")
        return

    # 1. 获取断点信息
    finished_urls, last_doc_num = get_finished_urls()

    # 2. 读取原始 URL 列表
    print("📖 正在读取输入文件...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    # 3. 生成任务列表 (过滤掉已完成的)
    print("⚙️ 正在生成任务队列...")
    tasks = []
    # 这里的 unique_urls 用于处理本次输入文件内部的重复
    # 同时也要和 finished_urls 做比对
    seen_in_this_run = set()
    
    doc_counter = last_doc_num + 1 # ID 接着上次的继续

    for item in raw_list:
        raw_url = item.get("url")
        if not raw_url: continue

        norm_url = normalize_url(raw_url)
        
        # 如果已经爬过 (断点)，或者本次运行任务里已经有了 (内部去重)
        if norm_url in finished_urls or norm_url in seen_in_this_run:
            continue
        
        seen_in_this_run.add(norm_url)
        
        tasks.append({
            "id": f"doc{doc_counter}",
            "url": raw_url
        })
        doc_counter += 1

    # 释放原始列表内存
    del raw_list 
    
    total_tasks = len(tasks)
    print(f"📊 任务统计：跳过 {len(finished_urls)} 条，剩余需爬取 {total_tasks} 条")
    
    if total_tasks == 0:
        print("🎉 所有任务已完成，无需爬取。")
        return

    print(f"🚀 启动并发爬取 (Workers={MAX_WORKERS})，结果将实时写入文件...")

    # 4. 边爬边写
    # 使用 'a' (append) 模式打开文件
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            futures = [executor.submit(fetch_and_process, task) for task in tasks]
            
            # 使用 tqdm 监控进度
            for future in tqdm(as_completed(futures), total=total_tasks, unit="页"):
                try:
                    result = future.result()
                    
                    # 核心修改：立即写入文件，不存内存
                    line = json.dumps(result, ensure_ascii=False)
                    f_out.write(line + "\n")
                    
                    # 可选：每写 10 条强制刷入硬盘，防止断电数据丢失太多
                    # f_out.flush() 
                    
                except Exception as e:
                    print(f"写入异常: {e}")

    print(f"\n🎉 爬取结束！数据已追加至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()