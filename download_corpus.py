# -*- coding: utf-8 -*-
"""按 arXiv ID 下载评测语料：30 篇 AI 经典论文 PDF。

用法（在仓库根目录执行）：
    python download_corpus.py            # 下载缺失的 PDF 到 my/corpus-ai-papers/
    python download_corpus.py --list     # 只打印 ID 清单，不下载
    python download_corpus.py --outdir ./my/corpus-ai-papers   # 自定义输出目录

说明：
- 文件命名为 "<arXiv ID>_<短标题>.pdf"，与评测集里记录的原始文件名一致。
- 已存在的文件自动跳过；单文件失败会重试，最终失败会明确报错并以非 0 退出。
- 语料为公开的 arXiv 论文，版权归原作者所有，请遵守 arXiv 使用条款。
"""

import sys
import time
from pathlib import Path

import httpx

# 文件名格式：<arXiv ID>_<短标题>.pdf（ID 即下划线前的数字部分）
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# 30 篇论文：(arXiv ID, 短标题)，短标题仅用于生成可读文件名
PAPERS: list[tuple[str, str]] = [
    ("1706.03762", "attention-is-all-you-need"),
    ("1810.04805", "bert"),
    ("2001.08361", "scaling-laws-kaplan"),
    ("2004.04906", "dpr"),
    ("2005.11401", "rag-lewis"),
    ("2104.09864", "rope-roformer"),
    ("2106.09685", "lora"),
    ("2201.11903", "chain-of-thought"),
    ("2203.02155", "instructgpt-rlhf"),
    ("2203.15556", "chinchilla"),
    ("2205.14135", "flash-attention"),
    ("2210.03629", "react"),
    ("2212.10496", "hyde"),
    ("2302.04761", "toolformer"),
    ("2302.13971", "llama"),
    ("2305.10601", "tree-of-thoughts"),
    ("2305.14314", "qlora"),
    ("2305.18290", "dpo"),
    ("2307.03172", "lost-in-the-middle"),
    ("2307.08691", "flash-attention-2"),
    ("2307.09288", "llama-2"),
    ("2310.11511", "self-rag"),
    ("2312.10997", "rag-survey"),
    ("2401.04088", "mixtral-moe"),
    ("2401.15884", "crag-corrective-rag"),
    ("2401.18059", "raptor"),
    ("2404.16130", "graphrag-microsoft"),
    ("2410.12813", "lightrag"),
    ("2412.19437", "deepseek-v3"),
    ("2501.12948", "deepseek-r1"),
]

DEFAULT_OUTDIR = "./my/corpus-ai-papers"
RETRIES = 3          # 单个文件的最大尝试次数
TIMEOUT = 120.0      # 单次请求超时（秒）


def filename_of(arxiv_id: str, slug: str) -> str:
    """拼接 <arXiv ID>_<短标题>.pdf。"""
    return f"{arxiv_id}_{slug}.pdf"


def list_ids() -> None:
    """打印 ID 清单（不联网）。"""
    print(f"共 {len(PAPERS)} 篇论文（arXiv ID -> 文件名）：")
    for i, (arxiv_id, slug) in enumerate(PAPERS, 1):
        print(f"{i:2d}. {arxiv_id}  {filename_of(arxiv_id, slug)}")


def download_one(client: httpx.Client, arxiv_id: str, slug: str, out_dir: Path) -> bool:
    """下载单个 PDF，成功返回 True；已存在则跳过并返回 True。"""
    dest = out_dir / filename_of(arxiv_id, slug)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] 已存在: {dest.name}")
        return True

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[get ] {arxiv_id} -> {dest.name} (第 {attempt} 次)")
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                # 先写临时文件，成功后再改名，避免半截文件被当成已完成
                tmp = dest.with_suffix(".pdf.part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                tmp.replace(dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"[ok  ] {dest.name}  ({size_mb:.1f} MB)")
            return True
        except Exception as e:  # 网络/HTTP 错误，重试
            last_err = e
            print(f"[warn] {arxiv_id} 第 {attempt} 次失败: {e}")
            if attempt < RETRIES:
                time.sleep(3 * attempt)

    print(f"[FAIL] {arxiv_id} 下载失败: {last_err}")
    return False


def main() -> None:
    if "--list" in sys.argv:
        list_ids()
        return

    out_dir = Path(DEFAULT_OUTDIR)
    if "--outdir" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--outdir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        for arxiv_id, slug in PAPERS:
            if download_one(client, arxiv_id, slug, out_dir):
                ok += 1

    print(f"\n完成 {ok}/{len(PAPERS)} 篇 -> {out_dir}")
    if ok < len(PAPERS):
        sys.exit(1)  # 有失败，明确以非 0 退出


if __name__ == "__main__":
    main()