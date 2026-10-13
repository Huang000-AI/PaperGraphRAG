# -*- coding: utf-8 -*-
"""汇总各实验的裁判判定 v2：完整通过率 + 平均得分点命中率。

用法（必须在仓库根目录运行）：
    python my/eval_summary.py

读 my/results/*.judged.json（v2 格式，含 verdict + hit_points / total_points）
口径：通过 = 二元判定（全部得分点命中 / 全部通过线满足）
输出两个核心指标：
  - 完整通过率（pass rate）：通过题数 / 总题数
  - 平均得分点命中率（hit rate）：仅可答题，Σ(hit/total) / 可答题数
"""

import json
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path("./my/results")
if "--outdir" in sys.argv:
    RESULTS = Path(sys.argv[sys.argv.index("--outdir") + 1])


def verdict_of(rec: dict) -> str:
    return (rec.get("judge") or {}).get("verdict", "未判")


def hit_rate(rec: dict) -> float | None:
    """可答题的得分点命中率；不可答题返回 None（不参与计算）。"""
    j = rec.get("judge") or {}
    if j.get("hit_points") is not None and j.get("total_points"):
        total = j["total_points"]
        return j["hit_points"] / total if total > 0 else None
    return None


def main():
    files = sorted(RESULTS.glob("eval_*.judged.json"))
    if not files:
        raise FileNotFoundError(f"未找到判定文件：{RESULTS}/eval_*.judged.json")

    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        recs = d["results"]
        counts = Counter(verdict_of(r) for r in recs)

        # 完整通过率（「通过」才算，幻觉否决计入不通过）
        n_pass = counts.get("通过", 0)
        n_total = len(recs)
        pass_rate = n_pass / n_total if n_total else 0.0

        # 幻觉否决题数（可答题中「点全中但因幻觉被毙」+ 不可答幻觉由通过线兜住）
        n_hallu_veto = counts.get("不通过（幻觉否决）", 0)

        # 平均得分点命中率（仅可答题）
        answerable = [r for r in recs if r.get("type") != "不可答"]
        hit_rates = [hit_rate(r) for r in answerable]
        hit_rates = [r for r in hit_rates if r is not None]
        avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0

        # 不可答题通过率
        unanswerable = [r for r in recs if r.get("type") == "不可答"]
        unans_pass = sum(1 for r in unanswerable if verdict_of(r) == "通过")
        unans_rate = unans_pass / len(unanswerable) if unanswerable else None

        rows.append({
            "name": f.name.replace(".judged.json", ""),
            "counts": counts,
            "pass_rate": pass_rate,
            "avg_hit": avg_hit,
            "unans_rate": unans_rate,
            "n_hallu_veto": n_hallu_veto,
            "n": n_total,
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "models": d.get("judge_models"),
        })

    # 汇总表
    print("实验 | 通过 | 不通过 | 幻觉否决 | 完整通过率 | 得分点命中率 | 不可答通过率")
    print("-" * 90)
    for r in rows:
        unans = f"{r['unans_rate']:.0%}" if r['unans_rate'] is not None else "N/A"
        print(
            f"{r['name']} | {r['counts'].get('通过', 0)} | "
            f"{r['counts'].get('不通过', 0)} | {r['n_hallu_veto']} | "
            f"{r['pass_rate']:.1%} | {r['avg_hit']:.1%} | {unans}"
        )

    # 逐题明细
    print("\n逐题明细：")
    for r in rows:
        print(
            f"\n[{r['name']}]  通过 {r['counts'].get('通过', 0)}/{r['n']} = {r['pass_rate']:.1%}"
            f"   命中率 {r['avg_hit']:.1%}   judge_models={r['models']}"
        )
        d = json.loads((RESULTS / f"{r['name']}.judged.json").read_text(encoding="utf-8"))
        for rec in d["results"]:
            j = rec.get("judge") or {}
            v = verdict_of(rec)
            if rec.get("type") == "不可答":
                print(f"  Q{rec['idx']:>2} [{rec['type']}] {v:<14} {j.get('reason', '')}")
            else:
                hit = j.get("hit_points", "?")
                total = j.get("total_points", "?")
                hallu = j.get("hallucination_claims") or []
                hallu_note = f"  幻觉{len(hallu)}处: {hallu[0].get('claim', '')[:30]}" if hallu else ""
                print(f"  Q{rec['idx']:>2} [{rec['type']}] {v:<14} 命中 {hit}/{total}  "
                      f"{j.get('reason', '')}{hallu_note}")


if __name__ == "__main__":
    main()
