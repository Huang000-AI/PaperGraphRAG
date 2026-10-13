# -*- coding: utf-8 -*-
"""评测基线：20 题逐题查询知识库，记录答案 + 检索到的 chunks。

默认配置：chunk 1200/100（建库时的设置）+ hybrid 模式 + 无 rerank。
用法（必须在仓库根目录运行）：
    python my/eval_baseline.py                          # 无 rerank
    python my/eval_baseline.py --rerank                 # 开 rerank
    python my/eval_baseline.py --rerank --name myexp    # 自定义实验名
    # 换知识库（如 2400 档）：必须同时指定配套的 testset（golden chunk 号按库重建）
    python my/eval_baseline.py \\
        --dir ./rag_eval_papers_2400 --testset ./my/testset_candidates_2400.json
    # 只重跑个别题（其余沿用已有结果）：--only 1,7
    python my/eval_baseline.py --rerank --name myexp --only 1,7
输出：my/results/<实验名>.json（默认实验名 eval_hybrid_norerank[_<库后缀>] / eval_hybrid_rerank）
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from openai import AsyncOpenAI

from lightrag import LightRAG, QueryParam
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
GLM_MODEL = _ENV["GLM_MODEL"]  # 模块级可变变量：改写即可切换生成模型

MIMO_BASE_URL = _ENV["MIMO_BASE_URL"]
MIMO_API_KEY = _ENV["MIMO_API_KEY"]
MIMO_MODEL = _ENV["MIMO_MODEL"]


def endpoint_of(model: str) -> tuple[str, str]:
    """按模型名前缀分发端点：mimo* 走 MiMo 网关，其余走 GLM。"""
    if model.lower().startswith("mimo"):
        return MIMO_BASE_URL, MIMO_API_KEY
    return GLM_BASE_URL, GLM_API_KEY

SILICONFLOW_BASE_URL = _ENV["SILICONFLOW_BASE_URL"].rstrip("/")
SILICONFLOW_API_KEY = _ENV["SILICONFLOW_API_KEY"]
SILICONFLOW_EMBED_MODEL = _ENV["SILICONFLOW_EMBED_MODEL"]
EMBEDDING_DIM = int(_ENV["EMBEDDING_DIM"])
RERANK_MODEL = _ENV["RERANK_MODEL"]


async def glm_chat_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """调用 GLM 生成回答。读模块级 GLM_MODEL，改写它即可换模型。

    直接复用 LightRAG 的 OpenAI 兼容绑定：它会自行处理 LightRAG 注入的
    hashing_kv（答案缓存）、response_format、流式与 COT，不会把内部 kwargs
    误传给 OpenAI SDK。
    """
    base_url, api_key = endpoint_of(GLM_MODEL)
    return await openai_complete_if_cache(
        model=GLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def siliconflow_embed(texts: list[str]) -> np.ndarray:
    """硅基流动 OpenAI 兼容 embeddings 端点（bge-m3 不支持 dimensions，不下发）。"""
    client = AsyncOpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    resp = await client.embeddings.create(model=SILICONFLOW_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


async def siliconflow_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    **kwargs,
) -> list[dict]:
    """调用硅基流动 /rerank，返回 LightRAG 约定格式 [{index, relevance_score}, ...]。"""
    payload: dict = {"model": RERANK_MODEL, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    async with httpx.AsyncClient(timeout=300) as http:
        resp = await http.post(
            f"{SILICONFLOW_BASE_URL}/rerank",
            headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"index": r["index"], "relevance_score": r["relevance_score"]}
        for r in data.get("results", [])
    ]


def strip_think(text: str | None) -> str | None:
    """剥离 <think>...</think> 推理段（GLM-Z1 等模型内联在 content 里返回）。"""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).split("<think>")[0].strip()

DEFAULT_WORK_DIR = "./rag_eval_papers"
WORK_DIR = DEFAULT_WORK_DIR
TESTSET = r"./my/testset_candidates_v1_1.json"
QUERY_MODE = "hybrid"

# ---- CLI 参数：--rerank 开 rerank，--name 指定实验名，--model 换生成模型 ----
USE_RERANK = "--rerank" in sys.argv
if "--dir" in sys.argv:
    WORK_DIR = sys.argv[sys.argv.index("--dir") + 1]
if "--testset" in sys.argv:
    TESTSET = sys.argv[sys.argv.index("--testset") + 1]
DEFAULT_MODEL = GLM_MODEL  # 先取 .env 里的默认模型，再允许被 --model 覆盖
if "--model" in sys.argv:
    QUERY_MODEL = sys.argv[sys.argv.index("--model") + 1]
else:
    QUERY_MODEL = DEFAULT_MODEL  # 默认沿用 my/.env 的 GLM_MODEL
GLM_MODEL = QUERY_MODEL  # glm_chat_complete 读模块级变量，覆盖即可生效
# 非默认知识库时给实验名加库后缀（rag_eval_papers_2400 -> "2400"），避免覆盖 1200 档结果
DIR_TAG = "" if WORK_DIR == DEFAULT_WORK_DIR else "_" + Path(WORK_DIR).name.replace("rag_eval_papers_", "")
if "--name" in sys.argv:
    EXPERIMENT = sys.argv[sys.argv.index("--name") + 1]
else:
    tag = "rerank" if USE_RERANK else "norerank"
    EXPERIMENT = (
        f"eval_{QUERY_MODEL}_{tag}" if QUERY_MODEL != DEFAULT_MODEL else f"eval_{QUERY_MODE}_{tag}"
    ) + DIR_TAG
OUT_DIR = Path("./my/results")
if "--outdir" in sys.argv:
    OUT_DIR = Path(sys.argv[sys.argv.index("--outdir") + 1])
OUT = OUT_DIR / f"{EXPERIMENT}.json"

# --only：只重跑指定题号（1 起，逗号分隔），其余题沿用已有结果文件中的记录
ONLY = None
if "--only" in sys.argv:
    ONLY = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}

# MiMo / GLM-Z1 是推理模型、生成慢，默认 240s（worker 480s）不够用，拉长到 600s
_m = QUERY_MODEL.lower()
LLM_TIMEOUT = 600 if (_m.startswith("mimo") or "z1" in _m) else 240

# chunk_id -> (file_path, chunk_order_index)，用于标记 golden chunk 命中
CHUNKS_KV = f"{WORK_DIR}/kv_store_text_chunks.json"


def load_chunk_index() -> dict[str, tuple[str, int]]:
    kv = json.load(open(CHUNKS_KV, encoding="utf-8"))
    return {
        cid: (
            (c.get("file_path") or "").replace("\\", "/").split("/")[-1],
            c.get("chunk_order_index", -1),
        )
        for cid, c in kv.items()
    }


def parse_golden(source: str) -> set[str]:
    """从 testset 的 source 字段解析 golden chunk 集合：{"file#chunkN", ...}"""
    out = set()
    for part in (source or "").split(";"):
        part = part.strip()
        if not part or part.startswith("N/A"):
            continue
        if "#chunk" in part:
            out.add(part)
    return out


async def main():
    testset = json.load(open(TESTSET, encoding="utf-8"))
    chunk_index = load_chunk_index()

    # --only 模式下，已有结果文件作为底，未重跑的题沿用旧记录
    prev_by_idx: dict[int, dict] = {}
    if ONLY:
        if not OUT.exists():
            raise FileNotFoundError(f"--only 需要已有结果作为底：{OUT} 不存在")
        for r in json.loads(OUT.read_text(encoding="utf-8"))["results"]:
            prev_by_idx[r["idx"]] = r

    # 无 rerank 时不传 rerank_model_func；有 rerank 时传入并在 QueryParam 打开
    rag = LightRAG(
        working_dir=WORK_DIR,
        embedding_func=siliconflow_embed,
        llm_model_func=glm_chat_complete,
        default_llm_timeout=LLM_TIMEOUT,
        **({"rerank_model_func": siliconflow_rerank} if USE_RERANK else {}),
    )
    await rag.initialize_storages()
    # 换模型必须绕开旧答案缓存：答案缓存键不含模型名，否则会重放旧模型的答案。
    # 每个实验用独立缓存实例（同类 JSON 存储、独立文件 kv_store_llm_response_cache_eval_<实验名>.json），
    # 原 kv_store_llm_response_cache.json（实体抽取缓存）完全不动，实验之间也互不串扰。
    eval_cache = rag.key_string_value_json_storage_cls(
        namespace=f"llm_response_cache_eval_{EXPERIMENT}",
        workspace=rag.workspace,
        global_config=rag.key_string_value_json_storage_cls.keywords["global_config"],
        embedding_func=rag.embedding_func,
    )
    await eval_cache.initialize()
    rag.llm_response_cache = eval_cache

    results = []

    def save() -> None:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "work_dir": WORK_DIR,
                    "mode": QUERY_MODE,
                    "rerank": USE_RERANK,
                    "query_model": QUERY_MODEL,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        for idx, item in enumerate(testset, 1):
            if ONLY and idx not in ONLY:
                if idx in prev_by_idx:
                    results.append(prev_by_idx[idx])
                continue
            q = item["question"]
            print(f"\n[{idx}/{len(testset)}] ({item['type']}) {q}")
            start = time.time()
            try:
                data = await rag.aquery_llm(
                    q, param=QueryParam(mode=QUERY_MODE, enable_rerank=USE_RERANK)
                )
                elapsed = round(time.time() - start, 1)

                status = data.get("status")
                answer = strip_think((data.get("llm_response") or {}).get("content"))
                llm_generated = (data.get("llm_response") or {}).get(
                    "llm_generated", False
                )
                chunks = (data.get("data") or {}).get("chunks") or []

                golden = parse_golden(item.get("source", ""))
                retrieved = []
                golden_hits = []
                for ch in chunks:
                    cid = ch.get("chunk_id", "")
                    fp, order = chunk_index.get(cid, ("?", -1))
                    tag = f"{fp}#chunk{order}"
                    hit = tag in golden
                    if hit:
                        golden_hits.append(tag)
                    retrieved.append(
                        {
                            "tag": tag,
                            "golden_hit": hit,
                            "content": ch.get("content", ""),
                        }
                    )

                print(f"    -> status={status}, chunks={len(retrieved)}, "
                      f"golden_hits={len(golden_hits)}, {elapsed}s")
                results.append(
                    {
                        "idx": idx,
                        "type": item["type"],
                        "question": q,
                        "ground_truth": item["ground_truth"],
                        "source": item.get("source", ""),
                        "answer": answer,
                        "llm_generated": llm_generated,
                        "status": status,
                        "n_chunks": len(retrieved),
                        "retrieved_tags": [r["tag"] for r in retrieved],
                        "golden_hits": golden_hits,
                        "retrieved": retrieved,
                        "elapsed_sec": elapsed,
                        "error": None,
                    }
                )
            except Exception as e:  # 单题失败不影响其余
                elapsed = round(time.time() - start, 1)
                print(f"    !!! 查询失败: {e}")
                results.append(
                    {
                        "idx": idx,
                        "type": item["type"],
                        "question": q,
                        "ground_truth": item["ground_truth"],
                        "source": item.get("source", ""),
                        "answer": None,
                        "llm_generated": False,
                        "status": "exception",
                        "n_chunks": 0,
                        "retrieved_tags": [],
                        "golden_hits": [],
                        "retrieved": [],
                        "elapsed_sec": elapsed,
                        "error": str(e),
                    }
                )
            # 每题落盘一次，中途断了也不丢
            save()
        # 循环结束后补写完整结果（--only 跳过的题不会触发逐题写入）
        save()
    finally:
        await rag.finalize_storages()
        if hasattr(eval_cache, "finalize"):
            await eval_cache.finalize()  # 评测缓存独立落盘（不在 finalize_storages 列表里）
        ok = sum(1 for r in results if r["error"] is None)
        print(f"\n完成 {ok}/{len(results)} 题，结果已写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
