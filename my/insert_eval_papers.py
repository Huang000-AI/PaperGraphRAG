"""把 my/corpus-ai-papers 下 30 篇 AI 论文批量入库，作为评测集知识库。

- 配置（GLM-4-Flash + BGE-M3）内联在本脚本，密钥/端点运行时从 my/.env 读取
- GLM-4-Flash 有限速：llm_model_max_async=2、max_parallel_insert=1
- 关 gleaning：entity_extract_max_gleaning=0（默认 1 会每 chunk 多抽一轮）
- 两个 chunk 档位各用独立目录，互不干扰：
  1200 -> ./rag_eval_papers（评测基线，20 题）；2400 -> ./rag_eval_papers_2400
"""

import asyncio
import glob
import os
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from pypdf import PdfReader
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs


def _load_env(path: Path) -> dict[str, str]:
    """解析 my/.env（KEY=VALUE，忽略 # 注释，去掉两侧引号）。"""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        env[key.strip()] = value
    return env


# 配置全部来自 my/.env，代码里不落任何密钥
_ENV = _load_env(Path(__file__).resolve().parent / ".env")

GLM_BASE_URL = _ENV["GLM_BASE_URL"]
GLM_API_KEY = _ENV["GLM_API_KEY"]
GLM_MODEL = _ENV["GLM_MODEL"]

SILICONFLOW_BASE_URL = _ENV["SILICONFLOW_BASE_URL"].rstrip("/")
SILICONFLOW_API_KEY = _ENV["SILICONFLOW_API_KEY"]
SILICONFLOW_EMBED_MODEL = _ENV["SILICONFLOW_EMBED_MODEL"]
EMBEDDING_DIM = int(_ENV["EMBEDDING_DIM"])


async def glm_chat_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """调用 GLM 生成回答。复用 LightRAG 的 OpenAI 兼容绑定，由其自行处理
    hashing_kv（答案缓存）、response_format、流式与 COT 等内部 kwargs。
    """
    return await openai_complete_if_cache(
        model=GLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=GLM_API_KEY,
        base_url=GLM_BASE_URL,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def siliconflow_embed(texts: list[str]) -> np.ndarray:
    """硅基流动 OpenAI 兼容 embeddings 端点（bge-m3 不支持 dimensions，不下发）。"""
    client = AsyncOpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    resp = await client.embeddings.create(model=SILICONFLOW_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

# 抽取模型实测记录（2026-09-21）：
# - glm-4-flash-250414：稳定，但账户并发硬上限 2（错误码 1302）
# - glm-4.7-flash：服务端过载，429 错误码 1305 连发，重试耗尽后整批 FAILED，不可用

CORPUS_DIR = "./my/corpus-ai-papers"
WORK_DIR = "./rag_eval_papers_2400"

# chunk=2400 档（1200 基线已建在 ./rag_eval_papers；要复现基线把下面两行改回 1200）
CHUNK_TOKEN_SIZE = 2400
CHUNK_OVERLAP_TOKEN_SIZE = 100

# GLM-4-Flash 限速保护（实测账户并发硬上限为 2：错误码 1302）
LLM_MAX_ASYNC = 2        # LLM 并发
MAX_PARALLEL_INSERT = 1  # 文档级串行


def extract_pdf_text(path: str) -> tuple[int, str]:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.replace("\x0c", "\n")  # 清掉 pypdf 的分页符
    return len(reader.pages), text


async def main():
    pdf_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"未找到 PDF: {CORPUS_DIR}")

    texts, file_paths = [], []
    for path in pdf_files:
        name = os.path.basename(path)
        pages, text = extract_pdf_text(path)
        if len(text) < 5000:
            print(f"[skip] {name}: 仅 {len(text)} 字符，可能是扫描件，需 OCR")
            continue
        texts.append(text)
        file_paths.append(name)
        print(f"[ok] {name}: {pages} 页, {len(text)} 字符")

    print(f"\n共 {len(texts)}/{len(pdf_files)} 篇论文待入库 -> {WORK_DIR}\n")

    rag = LightRAG(
        working_dir=WORK_DIR,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        # 默认 30s 会派生 60s 的 flush 超时，批量向量化跑不完会被掐断
        default_embedding_timeout=300,
        # GLM-4-Flash 限速：LLM 并发 2 + 文档串行
        llm_model_max_async=LLM_MAX_ASYNC,
        max_parallel_insert=MAX_PARALLEL_INSERT,
        # 关 gleaning：默认 1 会对每个 chunk 追加一轮抽取（等于抽两遍）
        entity_extract_max_gleaning=0,
        embedding_func=siliconflow_embed,
        llm_model_func=glm_chat_complete,
    )
    await rag.initialize_storages()
    try:
        await rag.ainsert(texts, file_paths=file_paths)
        print("\n全部论文入库完成")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
