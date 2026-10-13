# -*- coding: utf-8 -*-
"""一键启动 LightRAG Server + WebUI（复用 my/.env 里的密钥，不写进任何脚本）。

用法（任意目录）：
    python my/serve_webui.py [--port 9621]

启动后访问：
    WebUI:  http://localhost:9621/webui
    API 文档: http://localhost:9621/docs

说明：
- 知识库复用 ./rag_eval_papers（30 篇论文，chunk 1200），不会重建。
- LLM 走 MiMo（端点见 my/.env 的 MIMO_*）；Embedding / Rerank 走硅基流动。
- 服务端 rerank 绑定用 jina 兼容协议直连硅基流动 /rerank。
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env(path: Path) -> dict[str, str]:
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


def main() -> None:
    port = "9621"
    if "--port" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1]

    cfg = load_env(ENV_FILE)

    def get(key: str, default: str = "") -> str:
        return cfg.get(key) or os.environ.get(key) or default

    mimo_base = get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    sf_base = get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")

    server_env = {
        **os.environ,
        # 知识库（相对仓库根目录）
        "WORKING_DIR": "./rag_eval_papers",
        "INPUT_DIR": "./inputs",
        # LLM：OpenAI 兼容协议 -> MiMo
        "LLM_BINDING": "openai",
        "LLM_MODEL": get("MIMO_MODEL", "mimo-v2.6-flash"),
        "LLM_BINDING_HOST": mimo_base,
        "LLM_BINDING_API_KEY": get("MIMO_API_KEY"),
        # Embedding：OpenAI 兼容协议 -> 硅基流动 BGE-M3（1024 维，不下发 dimensions 参数）
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_MODEL": get("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3"),
        "EMBEDDING_BINDING_HOST": sf_base,
        "EMBEDDING_BINDING_API_KEY": get("SILICONFLOW_API_KEY"),
        "EMBEDDING_DIM": get("EMBEDDING_DIM", "1024"),
        "EMBEDDING_SEND_DIM": "false",
        # Rerank：jina 兼容协议 -> 硅基流动 /rerank
        "RERANK_BINDING": "jina",
        "RERANK_MODEL": get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        "RERANK_BINDING_HOST": f"{sf_base}/rerank",
        "RERANK_BINDING_API_KEY": get("SILICONFLOW_API_KEY"),
        # 服务
        "HOST": "127.0.0.1",
        "PORT": port,
        "WEBUI_TITLE": "PaperGraphRAG",
        "WEBUI_DESCRIPTION": "学术论文知识图谱问答系统",
    }

    if not server_env["LLM_BINDING_API_KEY"]:
        sys.exit("缺少 MIMO_API_KEY，请检查 my/.env")
    if not server_env["EMBEDDING_BINDING_API_KEY"]:
        sys.exit("缺少 SILICONFLOW_API_KEY，请检查 my/.env")

    print(f"知识库目录: {REPO_ROOT / 'rag_eval_papers'}")
    print(f"WebUI:      http://localhost:{port}/webui")
    print(f"API 文档:   http://localhost:{port}/docs")
    print("按 Ctrl+C 停止服务\n")

    subprocess.run(
        [sys.executable, "-m", "lightrag.api.lightrag_server"],
        cwd=REPO_ROOT,
        env=server_env,
        check=True,
    )


if __name__ == "__main__":
    main()
