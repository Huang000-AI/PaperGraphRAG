# -*- coding: utf-8 -*-
"""实验二·检索端内容级指标：平均检索 chunk 数 + golden 命中率 + golden 正文覆盖率。

为什么需要「内容级」指标：换 chunk 尺寸后，golden chunk 的编号与数量都会变，
单看「命中几个 golden chunk」会被切块尺寸扭曲。因此本脚本以 golden 正文的
**字符覆盖**为准，与切块尺寸无关。

口径定义（固定，保证可复现）：
1. 平均检索 chunk 数 = 各题 `n_chunks` 的均值。
2. golden 命中率 = Σ(该题命中的 golden 条目数) / Σ(该题的 golden 条目数)，条目级统计。
3. golden 正文覆盖率：取该题 golden chunk 的正文拼接（从知识库 `kv_store_text_chunks.json`
   按 `file#chunkN` 定位），以 **win=200 字窗口、stride=100** 滑窗切分；若某窗口完整
   出现在该题**检索到的任一 chunk 正文**中，记为已覆盖。单题覆盖率 = 已覆盖窗口数 /
   总窗口数，总体覆盖率 = 各题覆盖率的平均。
   （窗口取完整子串匹配，两侧空白差异按原样比较；不做归一化。）

用法（仓库根目录执行；--kb 指向**建该结果时所用的知识库目录**）：
    # chunk 1200 档（库目录 ./rag_eval_papers，结果目录 ./my/results1）
    python my/eval_coverage.py --kb ./rag_eval_papers --outdir ./my/results1 \\
        --names eval_mimo-v2.6-flash_norerank eval_mimo-v2.6-flash_rerank

    # chunk 2400 档（库目录 ./rag_eval_papers_2400，结果目录 ./my/results）
    python my/eval_coverage.py --kb ./rag_eval_papers_2400 --outdir ./my/results \\
        --names eval_mimo-v2.6-flash_norerank_2400 eval_mimo-v2.6-flash_rerank_2400
"""

import json
import statistics
import sys
from pathlib import Path


def _list_arg(flag: str) -> list[str]:
    """取 --flag 之后的连续非 `--` 参数作为列表。"""
    if flag not in sys.argv:
        return []
    out: list[str] = []
    for x in sys.argv[sys.argv.index(flag) + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def _opt_arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


KB_DIR = _opt_arg("--kb", "./rag_eval_papers")
OUT_DIR = Path(_opt_arg("--outdir", "./my/results"))
NAMES = _list_arg("--names")
WIN = int(_opt_arg("--win", "200"))
STRIDE = int(_opt_arg("--stride", "100"))

if not NAMES:
    raise SystemExit("需要 --names <实验名...>（可多个）")


def load_kb_chunks(kb_dir: str) -> dict[str, str]:
    """tag(`file#chunkN`) -> 正文。"""
    kv = json.loads(
        (Path(kb_dir) / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    m: dict[str, str] = {}
    for c in kv.values():
        fp = (c.get("file_path") or "").replace("\\", "/").split("/")[-1]
        m[f"{fp}#chunk{c.get('chunk_order_index', -1)}"] = c.get("content", "")
    return m


def golden_tags(source: str) -> list[str]:
    return [p.strip() for p in (source or "").split(";") if "#chunk" in p]


def question_windows(text: str) -> list[str]:
    if len(text) < WIN:
        return [text] if text else []
    return [text[i:i + WIN] for i in range(0, len(text) - WIN + 1, STRIDE)]


def evaluate_one(kb: dict[str, str], name: str) -> dict:
    d = json.loads((OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    recs = d["results"]

    n_chunks = [r["n_chunks"] for r in recs]

    hit_total = gold_total = 0          # 条目级命中
    per_q_cov: list[float] = []         # 逐题覆盖率
    for r in recs:
        tags = golden_tags(r.get("source", ""))
        if not tags:
            continue
        got = set(r.get("retrieved_tags") or [])
        hit_total += sum(1 for t in tags if t in got)
        gold_total += len(tags)

        gold_text = "".join(kb.get(t, "") for t in tags)
        ctx = " ".join((c.get("content") or "") for c in (r.get("retrieved") or []))
        wins = question_windows(gold_text)
        if wins:
            covered = sum(1 for w in wins if w in ctx)
            per_q_cov.append(covered / len(wins))

    return {
        "name": name,
        "n": len(recs),
        "avg_chunks": statistics.mean(n_chunks) if n_chunks else 0.0,
        "golden_hit_rate": hit_total / gold_total if gold_total else 0.0,
        "coverage": statistics.mean(per_q_cov) if per_q_cov else 0.0,
        "n_goldens": gold_total,
        "n_cov_questions": len(per_q_cov),
    }


def main() -> None:
    kb = load_kb_chunks(KB_DIR)
    print(f"KB = {KB_DIR}   (win={WIN}, stride={STRIDE})")
    print(f"{'实验':<42} {'平均检索chunk数':>12} {'golden命中率':>12} {'正文覆盖率':>10}")
    print("-" * 82)
    for name in NAMES:
        s = evaluate_one(kb, name)
        print(
            f"{s['name']:<42} {s['avg_chunks']:>12.2f} "
            f"{s['golden_hit_rate']:>11.1%} {s['coverage']:>10.1%}"
            f"   （{s['n_goldens']} 个 golden 条目 / {s['n_cov_questions']} 题）"
        )


if __name__ == "__main__":
    main()
