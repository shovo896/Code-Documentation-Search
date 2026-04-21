import sys
import secrets
import time
from pathlib import Path

import gradio as gr
from gradio import networking
from gradio.tunneling import CURRENT_TUNNELS

from ingest import ingest_repository, normalize_repo_url
from retriever import get_qa_chain


chain_cache = {}
LIVE_URL_FILE = Path("gradio_live_url.txt")


def get_cached_chain(repo_info):
    namespace = repo_info["namespace"]
    if namespace not in chain_cache:
        chain_cache[namespace] = get_qa_chain(
            namespace=namespace,
            repo_url=repo_info["repo_url"],
            branch=repo_info["branch"],
        )
    return chain_cache[namespace]


def format_status(repo_info):
    return (
        f"Loaded repository: {repo_info['repo_url']}\n"
        f"Branch: {repo_info['branch']}\n"
        f"Files indexed: {repo_info['files']}\n"
        f"Chunks indexed: {repo_info['chunks']}"
    )


def load_repository(repo_url, branch):
    try:
        repo_url = normalize_repo_url(repo_url)
        branch = (branch or "").strip()
        repo_info = ingest_repository(repo_url, branch)
        chain_cache.pop(repo_info["namespace"], None)
        get_cached_chain(repo_info)
        return repo_info, format_status(repo_info), "", ""
    except Exception as e:
        print(f"Repository load error: {e}", file=sys.stderr)
        return None, f"Error: {e}", "", ""


def answer_query(repo_url, branch, question, repo_state):
    if not repo_url.strip():
        return repo_state, "Please enter a GitHub repository URL.", "", ""

    if not question.strip():
        return repo_state, "Please enter a question.", "", ""

    try:
        repo_url = normalize_repo_url(repo_url)
        branch = (branch or "").strip()
        state_matches = (
            repo_state
            and repo_state.get("repo_url") == repo_url
            and (not branch or repo_state.get("branch") == branch)
        )

        if not state_matches:
            repo_state = ingest_repository(repo_url, branch)
            chain_cache.pop(repo_state["namespace"], None)

        chain = get_cached_chain(repo_state)
        result = chain.invoke({"query": question})
        answer = result["result"]

        sources = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "unknown")
            sources.add(source)

        source_text = "\n".join(sorted(sources))
        return repo_state, format_status(repo_state), answer, source_text

    except Exception as e:
        print(f"Query error: {e}", file=sys.stderr)
        return repo_state, f"Error: {e}", "", ""


with gr.Blocks(title="Code Doc Search") as demo:
    repo_state = gr.State(None)

    gr.Markdown("Code Documentation Search")
    gr.Markdown(
        "Paste any public GitHub repository URL, then ask questions about that codebase. "
        "Answers are always returned in English."
    )

    repo_url = gr.Textbox(
        label="GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        lines=1,
    )

    branch = gr.Textbox(
        label="Branch",
        value="",
        placeholder="Leave empty to use the repository default branch",
        lines=1,
    )

    load_btn = gr.Button("Load Repository", variant="secondary")
    status_box = gr.Textbox(label="Repository Status", lines=4, interactive=False)

    query = gr.Textbox(
        label="Your Question",
        placeholder="e.g. How does authentication work in this repository?",
        lines=2,
    )

    search_btn = gr.Button("Search", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=10)
        source_box = gr.Textbox(label="Sources", lines=10)

    load_btn.click(
        fn=load_repository,
        inputs=[repo_url, branch],
        outputs=[repo_state, status_box, answer_box, source_box],
    )

    search_btn.click(
        fn=answer_query,
        inputs=[repo_url, branch, query, repo_state],
        outputs=[repo_state, status_box, answer_box, source_box],
    )


def keep_share_tunnel_alive():
    while True:
        try:
            token = secrets.token_urlsafe(32)
            url = networking.setup_tunnel(
                local_host="127.0.0.1",
                local_port=7860,
                share_token=token,
                share_server_address=None,
                share_server_tls_certificate=None,
            )
            LIVE_URL_FILE.write_text(url, encoding="utf-8")
            print(f"* Running on public URL: {url}", flush=True)

            tunnel = CURRENT_TUNNELS[-1]
            while tunnel.proc is not None and tunnel.proc.poll() is None:
                time.sleep(5)
            print("Public tunnel stopped. Restarting...", flush=True)
        except Exception as e:
            print(f"Public tunnel error: {e}", file=sys.stderr, flush=True)
        time.sleep(5)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        prevent_thread_lock=False,
        debug=True,
        show_error=True,
    )
