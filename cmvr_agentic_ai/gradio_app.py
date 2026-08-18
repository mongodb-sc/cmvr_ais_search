"""Gradio chatbot UI for the CMVR/AIS agentic test-finder.

Run:  python cmvr_agentic_ai/gradio_app.py

Streams each tool call as a collapsible reasoning step, then the final cited
answer, in a chat-style interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import os
import queue
import threading

# Silence transformers' lazy `__path__` warnings (pulled in via voyageai). Must
# run before the agent modules import transformers transitively.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr

from agent.loop import run_agent, summarize_tool_call

INTRO = (
    "# 🚌 CMVR / AIS Agentic Test Finder\n"
    "Ask which tests or requirements apply to a vehicle. The agent searches the "
    "**Central Motor Vehicle Rules** first, then the **Automotive Industry "
    "Standards** clauses, and answers with citations. Each tool call shows up as "
    "a collapsible reasoning step."
)

_ARCH_SVG = (Path(__file__).resolve().parent / "assets" / "architecture.svg").read_text(
    encoding="utf-8"
)

_ARCH_HTML = (
    "<div style='width:100%; max-width:1720px; margin:0 auto; padding:8px 4px; "
    "background:#ffffff; border-radius:10px;'>"
    + _ARCH_SVG
    + "</div>"
)

ARCH_NOTES = (
    "### How it works\n"
    "1. **Agent loop** ([agent/loop.py](agent/loop.py)) drives a raw Claude "
    "Messages API tool-use loop through the **Grove gateway** "
    "([agent/llm_client.py](agent/llm_client.py), httpx, `api-key` header).\n"
    "2. **`cmvr_search`** always runs first: hybrid search over `cmvr_rules`, "
    "returning matched rules and the union of referenced AIS codes.\n"
    "3. **`ais_search`** runs next (iterative, hard-capped at 2 calls). The AIS "
    "codes act as a **hard `AIS_id` pre-filter**; results include "
    "`further_ais_refs` so the agent can loop.\n"
    "4. Both tools share one **hybrid retrieval** pipeline "
    "([search/hybrid.py](search/hybrid.py)): MongoDB **`$rankFusion`** fusing "
    "`$vectorSearch` (voyage-4-large, 1024d) + Atlas `$search` full-text, then "
    "**Voyage `rerank-2.5`**.\n"
    "5. After the AIS budget is spent, the tools are dropped so the model writes "
    "a **final answer with citations** back to rule numbers / AIS clauses."
)

EXAMPLES = [
    "Find all tests for a bus.",
    "What are the braking requirements for M3 category vehicles?",
    "Which AIS standards apply to the body building and approval of buses?",
    "What tests are needed for a new cooling system installed in a bus?",
]


def add_user(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Append the user's message and clear the input box."""
    history = history or []
    if (user_message or "").strip():
        history = history + [{"role": "user", "content": user_message.strip()}]
    return "", history


def agent_reply(history: list[dict]):
    """Run the agent for the last user message, streaming reasoning steps."""
    if not history or history[-1].get("role") != "user":
        yield history
        return

    user_message = history[-1]["content"]
    events: queue.Queue = queue.Queue()

    def on_tool_call(call) -> None:
        events.put(("tool", call))

    def worker() -> None:
        try:
            run = run_agent(user_message, on_tool_call=on_tool_call, verbose=True)
            events.put(("done", run))
        except Exception as error:  # surface failures into the chat
            events.put(("error", error))

    threading.Thread(target=worker, daemon=True).start()

    # A pending status bubble while the first tool call is prepared.
    history = history + [
        {
            "role": "assistant",
            "content": "Searching CMVR and AIS collections…",
            "metadata": {"title": "🔎 Working", "status": "pending"},
        }
    ]
    yield history
    status_index = len(history) - 1

    step = 0
    while True:
        kind, payload = events.get()
        if kind == "tool":
            step += 1
            history[status_index] = {
                "role": "assistant",
                "content": summarize_tool_call(payload, step),
                "metadata": {"title": f"🔧 Step {step}: {payload.name}", "status": "done"},
            }
            # Add a fresh pending bubble for the next step.
            history = history + [
                {
                    "role": "assistant",
                    "content": "Thinking…",
                    "metadata": {"title": "🔎 Working", "status": "pending"},
                }
            ]
            status_index = len(history) - 1
            yield history
        elif kind == "done":
            answer = payload.answer or "_No answer produced._"
            caption = (
                f"_Stopped: {payload.stopped_reason} · "
                f"{len(payload.tool_calls)} tool call(s) · {payload.turns} turn(s)_"
            )
            history[status_index] = {"role": "assistant", "content": f"{answer}\n\n{caption}"}
            yield history
            return
        elif kind == "error":
            history[status_index] = {
                "role": "assistant",
                "content": f"⚠️ **{type(payload).__name__}**: {payload}",
                "metadata": {"title": "Error", "status": "done"},
            }
            yield history
            return


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CMVR/AIS Test Finder") as demo:
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("💬 Chatbot"):
                chatbot = gr.Chatbot(
                    height=580,
                    label="Agent",
                    resizable=True,
                    line_breaks=True,
                    avatar_images=(None, None),
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="e.g. Find all tests for a bus.",
                        scale=8,
                        show_label=False,
                        autofocus=True,
                        submit_btn=True,
                    )
                    clear = gr.Button("Clear", scale=1)

                gr.Examples(EXAMPLES, inputs=msg, label="Try an example")

                msg.submit(
                    add_user, [msg, chatbot], [msg, chatbot], queue=False
                ).then(agent_reply, chatbot, chatbot)
                clear.click(lambda: [], None, chatbot, queue=False)

            with gr.Tab("🗺️ Architecture"):
                gr.HTML(_ARCH_HTML)
                gr.Markdown(ARCH_NOTES)

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        theme=gr.themes.Soft(),
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
