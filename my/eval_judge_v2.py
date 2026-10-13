# -*- coding: utf-8 -*-
"""逐题裁判 v2：逐得分点核对命中/未命中，输出结构化 verdict。

与 v1 的核心区别：
1. 读取 v1.1 测试集的 scoring_points / pass_criteria 字段
2. 可答题：裁判逐得分点核对，输出 JSON（hit_points / total_points / hit_list / reason）
3. 不可答题：二元判定（通过 = 全部通过线满足，否则不通过）
4. verdict 词统一为「通过 / 不通过」

用法：
    python my/eval_judge_v2.py [实验名] [--only 18,19,20] [--testset ./my/testset_candidates_2400.json]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI


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

MIMO_BASE_URL = _ENV["MIMO_BASE_URL"]
MIMO_API_KEY = _ENV["MIMO_API_KEY"]
MIMO_MODEL = _ENV["MIMO_MODEL"]

GLM_BASE_URL = _ENV["GLM_BASE_URL"]
GLM_API_KEY = _ENV["GLM_API_KEY"]
GLM_MODEL = _ENV["GLM_MODEL"]


def endpoint_of(model: str) -> tuple[str, str]:
    """按模型名前缀分发端点，返回 (base_url, api_key)。"""
    if model.lower().startswith("mimo"):
        return MIMO_BASE_URL, MIMO_API_KEY
    return GLM_BASE_URL, GLM_API_KEY


EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "eval_hybrid_norerank"
OUT_DIR = Path("./my/results")
if "--outdir" in sys.argv:
    OUT_DIR = Path(sys.argv[sys.argv.index("--outdir") + 1])
IN = OUT_DIR / f"{EXPERIMENT}.json"
OUT = OUT_DIR / f"{EXPERIMENT}.judged.json"
TESTSET = Path("./my/testset_candidates_v1_1.json")
if "--testset" in sys.argv:
    TESTSET = Path(sys.argv[sys.argv.index("--testset") + 1])
MODEL = MIMO_MODEL
CONCURRENCY = 4

ONLY = None
if "--only" in sys.argv:
    ONLY = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}

_BASE_URL, _API_KEY = endpoint_of(MODEL)
client = AsyncOpenAI(base_url=_BASE_URL, api_key=_API_KEY, timeout=300)
sem = asyncio.Semaphore(CONCURRENCY)


# ---- prompts ----

JUDGE_ANSWERABLE = (
    "你是严格的评分员。给定「问题」「标准答案（含得分点列表）」「RAG 系统的回答」。\n"
    "任务一：逐得分点核对 RAG 回答是否命中该得分点。\n"
    "任务二：幻觉检查。检查回答中是否存在以下两类内容：\n"
    "  a. 与标准答案矛盾的内容（数字抄错、归属说反、事实冲突）；\n"
    "  b. 编造的不存在的实体或文献（虚构的模型名、论文名、引文、数据）。\n"
    "注意：与标准答案不矛盾、非编造的正确补充细节不算幻觉，不要列入。\n"
    "只输出合法的 JSON，不要任何 markdown 代码块标记，不要任何额外说明文字：\n"
    "{\"hit_points\": 命中数, \"total_points\": 总得分点数, "
    "\"hit_list\": [{\"point\": \"得分点原文\", \"hit\": true/false, \"reason\": \"简要理由\"}, ...], "
    "\"hallucination_claims\": [{\"claim\": \"问题内容摘录\", \"type\": \"与标准答案矛盾/编造实体或文献\", "
    "\"reason\": \"简要理由\"}, ...], "
    "\"overall_reason\": \"不超过 40 字的总结理由\"}\n"
    "若无幻觉内容，hallucination_claims 输出空数组 []。"
)

JUDGE_UNANSWERABLE = (
    "你是严格的评分员。这是一道「库外不可答」题，知识库中本无答案。\n"
    "给定「问题」「通过线标准」「RAG 系统的回答」。\n"
    "请逐条核对 RAG 回答是否满足全部通过线。\n"
    "只输出合法的 JSON，不要任何 markdown 代码块标记，不要任何额外说明文字：\n"
    "{\"passed\": true/false, "
    "\"line_checks\": [{\"line\": \"通过线原文\", \"passed\": true/false, \"reason\": \"简要理由\"}, ...], "
    "\"overall_reason\": \"不超过 40 字的总结理由\"}"
)


def pick_prompt_and_fields(rec: dict, ts_item: dict) -> tuple[str, str, list[str]]:
    """返回 (prompt_type, system_prompt, points_or_lines)。"""
    if rec["type"] == "不可答":
        lines = []
        # 从 GT 里提取通过线（【通过线】之后的编号项）
        gt = ts_item.get("ground_truth", "")
        for m in re.finditer(r"(\d+)\.\s+(.+)", gt):
            lines.append(m.group(2).strip())
        return "unanswerable", JUDGE_UNANSWERABLE, lines

    points = ts_item.get("scoring_points", [])
    if not points:
        # fallback：把整个 GT 当单一点
        points = [ts_item.get("ground_truth", "")]
    return "answerable", JUDGE_ANSWERABLE, points


async def chat(system: str, user: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=3000,
                )
                return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                return f"[调用失败] {e}"
            await asyncio.sleep(3 * (attempt + 1))
    return "[调用失败]"


def _extract_json(text: str) -> dict | None:
    """从裁判输出中提取 JSON（去掉 markdown 代码块等包装）。"""
    text = text.strip()
    # 去掉 markdown 代码块
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fallback_parse(text: str, kind: str, points: list[str]) -> dict:
    """当 JSON 解析失败时，做兜底解析。"""
    text = text.strip()
    if kind == "answerable":
        # 尝试用正则找命中数
        m = re.search(r"(\d+)\s*/\s*(\d+)", text)
        hit = int(m.group(1)) if m else 0
        total = int(m.group(2)) if m else len(points)
        return {
            "hit_points": hit,
            "total_points": total,
            "hit_list": [],
            "hallucination_claims": [],
            "overall_reason": text[:80],
        }
    else:
        passed = "通过" in text or "passed\": true" in text or "满足" in text
        return {
            "passed": passed,
            "line_checks": [],
            "overall_reason": text[:80],
        }


async def judge_one(rec: dict, ts_item: dict) -> dict:
    if rec.get("error") or not rec.get("answer"):
        return {
            "verdict": "不通过",
            "hit_points": 0,
            "total_points": 0,
            "hit_list": [],
            "hallucination_claims": [],
            "reason": f"系统失败: {rec.get('error') or '无回答'}",
            "model": "-",
            "prompt": "-",
        }

    kind, sys_p, points = pick_prompt_and_fields(rec, ts_item)
    user_parts = [f"问题：{rec['question']}"]

    if kind == "answerable":
        user_parts.append(f"标准答案：{ts_item.get('ground_truth', '')}")
        user_parts.append(f"得分点列表（共 {len(points)} 个）：")
        for i, p in enumerate(points, 1):
            user_parts.append(f"  {i}. {p}")
    else:
        user_parts.append(f"通过线标准：")
        for i, p in enumerate(points, 1):
            user_parts.append(f"  {i}. {p}")

    user_parts.append(f"RAG 回答：{rec['answer']}")
    user = "\n".join(user_parts)

    text = await chat(sys_p, user)
    data = _extract_json(text)
    if data is None:
        data = _fallback_parse(text, kind, points)

    if kind == "answerable":
        hit = int(data.get("hit_points", 0))
        total = int(data.get("total_points", len(points)))
        hallucinations = data.get("hallucination_claims") or []
        # 判定：得分点全中 且 无幻觉 才通过；幻觉一票否决
        if hit >= total and not hallucinations:
            verdict = "通过"
        elif hallucinations:
            verdict = "不通过（幻觉否决）"
        else:
            verdict = "不通过"
        return {
            "verdict": verdict,
            "hit_points": hit,
            "total_points": total,
            "hit_list": data.get("hit_list", []),
            "hallucination_claims": hallucinations,
            "reason": data.get("overall_reason", data.get("reason", "")),
            "model": MODEL,
            "prompt": kind,
            "raw_response": text[:500],
        }
    else:
        passed = bool(data.get("passed", False))
        verdict = "通过" if passed else "不通过"
        return {
            "verdict": verdict,
            "passed": passed,
            "line_checks": data.get("line_checks", []),
            "reason": data.get("overall_reason", data.get("reason", "")),
            "model": MODEL,
            "prompt": kind,
            "raw_response": text[:500],
        }


async def main():
    if not IN.exists():
        raise FileNotFoundError(f"输入不存在: {IN}")
    if not TESTSET.exists():
        raise FileNotFoundError(f"测试集不存在: {TESTSET}")

    d = json.loads(IN.read_text(encoding="utf-8"))
    results = d["results"]

    # 加载 v1.1 测试集，按 idx 索引
    ts_data = json.loads(TESTSET.read_text(encoding="utf-8"))
    ts_by_idx = {i + 1: item for i, item in enumerate(ts_data)}

    # --only 模式：已有 .judged.json 为底
    prev = {}
    if ONLY:
        if not OUT.exists():
            raise FileNotFoundError(f"--only 需要已有判定作为底：{OUT} 不存在")
        for r in json.loads(OUT.read_text(encoding="utf-8"))["results"]:
            if r.get("judge"):
                prev[r["idx"]] = r["judge"]

    for rec in results:
        idx = rec["idx"]
        if ONLY and idx not in ONLY:
            rec["judge"] = prev.get(idx) or {
                "verdict": "未判", "reason": "本次未重判且无历史判定",
                "model": "-", "prompt": "-",
            }
            continue

        ts_item = ts_by_idx.get(idx, {})
        rec["judge"] = await judge_one(rec, ts_item)
        j = rec["judge"]
        gh = len(rec.get("golden_hits", []))
        print(
            f"Q{idx:2d} [{rec['type']}] {j['verdict']:<6s} "
            f"golden命中 {gh}  {j['reason']}"
        )

    d["judged_at"] = True
    models: dict[str, int] = {}
    for rec in results:
        m = (rec.get("judge") or {}).get("model", "-")
        models[m] = models.get(m, 0) + 1
    d["judge_models"] = models
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n裁判模型分布: {models}")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
