# -*- coding: utf-8 -*-
"""RAGAS 四指标评测：faithfulness / answer_relevancy / context_precision / context_recall。

前置：先跑 my/eval_baseline.py 产出 my/results/<实验名>.json
用法：python my/eval_ragas.py [实验名]（默认 eval_hybrid_norerank）
- 17 道可答题进 RAGAS；3 道不可答题不进（context_recall 对拒答题无意义），单独统计拒答行为。
- 并发压到 2（保守值，避免 judge 端限流）。
输出：打印汇总 + my/results/<实验名>.ragas.json（及 .per_question.json）
"""

import asyncio
import json
import sys
from pathlib import Path

from ragas import evaluate
from ragas.run_config import RunConfig

from ragas_config import eval_metrics, evaluator_embeddings, evaluator_llm

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "eval_hybrid_norerank"
OUT_DIR = Path("./my/results")
if "--outdir" in sys.argv:
    OUT_DIR = Path(sys.argv[sys.argv.index("--outdir") + 1])
IN = OUT_DIR / f"{EXPERIMENT}.json"
OUT = OUT_DIR / f"{EXPERIMENT}.ragas.json"


def build_rows(results: list[dict]) -> list[dict]:
    rows = []
    for rec in results:
        if rec["type"] == "不可答" or rec.get("error") or not rec.get("answer"):
            continue
        contexts = [c["content"] for c in rec.get("retrieved", [])]
        rows.append(
            {
                "user_input": rec["question"],
                "retrieved_contexts": contexts,
                "response": rec["answer"],
                "reference": rec["ground_truth"],
            }
        )
    return rows


async def main():
    d = json.loads(IN.read_text(encoding="utf-8"))
    results = d["results"]

    # 不可答题单独统计拒答行为
    unans = [r for r in results if r["type"] == "不可答"]
    unans_report = []
    for r in unans:
        a = (r.get("answer") or "").strip()
        refused = any(
            k in a for k in ("无法", "不知道", "未公开", "没有相关", "库中", "无法确定", "抱歉")
        ) or len(a) < 120
        unans_report.append(
            {"idx": r["idx"], "question": r["question"], "answer_head": a[:150],
             "refused_guess": refused}
        )

    rows = build_rows(results)
    print(f"进入 RAGAS 的可答题数: {len(rows)}（不可答 {len(unans)} 题单独统计）")

    ds = EvaluationDataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=eval_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=2, max_retries=5, timeout=600),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    summary = {
        "n_samples": len(rows),
        "scores": {m: float(df[m].mean()) for m in (
            "faithfulness", "answer_relevancy", "context_precision", "context_recall")
            if m in df.columns},
        "unanswerable": unans_report,
    }
    print("\n===== RAGAS 四指标（17 题均值）=====")
    for k, v in summary["scores"].items():
        print(f"  {k:20s} {v:.4f}")
    print("\n===== 不可答题拒答检查 =====")
    for u in unans_report:
        print(f"  Q{u['idx']} 拒答? {u['refused_guess']}  {u['answer_head'][:60]}")

    OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 明细也留一份，方便逐题看
    per_q = []
    for i, row in enumerate(rows):
        per_q.append({
            "question": row["user_input"],
            **{m: (float(df[m].iloc[i]) if m in df.columns else None)
               for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
        })
    OUT.with_suffix(".per_question.json").write_text(
        json.dumps(per_q, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {OUT}")


from ragas import EvaluationDataset  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
