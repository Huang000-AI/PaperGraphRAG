"""RAGAS 评测配置：judge LLM + judge embedding。

- judge LLM: my/.env 里的 MIMO_MODEL（小米 MiMo，OpenAI 兼容端点）
- judge embedding: 硅基流动 BGE-M3（与 LightRAG 知识库同款 embedding）

用法（评测脚本里）:
    from ragas_config import evaluator_llm, evaluator_embeddings, eval_metrics
    from ragas import evaluate
    result = evaluate(dataset, metrics=eval_metrics, llm=evaluator_llm,
                      embeddings=evaluator_embeddings)

直接运行本文件做连通性自检。
"""

from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


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

MIMO_API_KEY = _ENV["MIMO_API_KEY"]
MIMO_BASE_URL = _ENV["MIMO_BASE_URL"]
MIMO_MODEL = _ENV["MIMO_MODEL"]

SILICONFLOW_API_KEY = _ENV["SILICONFLOW_API_KEY"]
SILICONFLOW_BASE_URL = _ENV["SILICONFLOW_BASE_URL"]
SILICONFLOW_EMBED_MODEL = _ENV["SILICONFLOW_EMBED_MODEL"]

# ---- judge LLM: 小米 MiMo（端点与密钥均来自 my/.env）----
JUDGE_LLM_MODEL = MIMO_MODEL

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=JUDGE_LLM_MODEL,
        base_url=MIMO_BASE_URL,
        api_key=MIMO_API_KEY,
        temperature=0,  # 裁判要稳定，不要创造性
        timeout=600,  # 推理模型 RAGAS 重 prompt 常超 2 分钟，放宽超时
    ),
    bypass_n=True,  # MiMo 端点不支持 n>1 参数，改为发 n 次独立请求
)

# ---- judge embedding: 硅基流动 BGE-M3 ----
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model=SILICONFLOW_EMBED_MODEL,
        base_url=SILICONFLOW_BASE_URL,
        api_key=SILICONFLOW_API_KEY,
        check_embedding_ctx_length=False,  # 非 OpenAI 模型，跳过 tiktoken 分块
    )
)

# ---- 四个核心指标 ----
eval_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]


async def sanity_check():
    """连通性自检：judge LLM 一次对话 + judge embedding 一次向量化。"""
    from langchain_openai import ChatOpenAI as _C

    chat: _C = evaluator_llm.langchain_llm
    resp = await chat.ainvoke("只回复两个字：正常")
    print(f"[judge LLM] {JUDGE_LLM_MODEL} -> {resp.content!r}")

    emb = await evaluator_embeddings.embed_text("RAGAS evaluation sanity check")
    print(f"[judge embedding] {SILICONFLOW_EMBED_MODEL} -> {len(emb)} 维向量")


if __name__ == "__main__":
    import asyncio

    asyncio.run(sanity_check())
