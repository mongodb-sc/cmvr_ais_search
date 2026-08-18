"""Minimal Streamlit UI for the CMVR/AIS agentic test-finder.

Run:  streamlit run cmvr_agentic_ai/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import os

# Silence transformers' lazy `__path__` warnings. transformers is pulled in
# transitively via voyageai, and Streamlit's file-watcher walks every imported
# module's __path__, which re-triggers these warnings en masse. This must run
# before the agent modules import transformers transitively.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

import json

import streamlit as st

from agent.loop import run_agent, summarize_tool_call

st.set_page_config(page_title="CMVR/AIS Test Finder", layout="wide")
st.title("CMVR / AIS Agentic Test Finder")
st.caption(
    "Ask which tests/requirements apply to a vehicle. The agent searches CMVR "
    "rules first, then loops over AIS clauses, and cites every source."
)

query = st.text_input(
    "Query",
    value="Identify every mandatory test requirement needed for approval of a new cooling system installed in buses having 16–32 passenger capacity.",
    placeholder="e.g. Find all tests for a bus.",
)

if st.button("Search", type="primary") and query.strip():
    step = {"n": 0}
    with st.status("Reasoning over CMVR and AIS collections…", expanded=True) as status:
        def on_tool_call(call) -> None:
            step["n"] += 1
            status.update(label=f"Step {step['n']}: {call.name}…")
            status.markdown(summarize_tool_call(call, step["n"]))

        try:
            run = run_agent(query.strip(), verbose=True, on_tool_call=on_tool_call)
        except Exception as error:  # keep the UI alive on failures
            status.update(label="Failed", state="error")
            st.error(f"{type(error).__name__}: {error}")
            st.stop()
        status.update(
            label=f"Done · {len(run.tool_calls)} tool call(s)", state="complete"
        )

    st.subheader("Answer")
    st.markdown(run.answer or "_No answer produced._")
    st.caption(f"Stopped: {run.stopped_reason} · {len(run.tool_calls)} tool call(s) · {run.turns} turn(s)")

    if run.tool_calls:
        st.subheader("Tool-call summary")
        for index, call in enumerate(run.tool_calls, start=1):
            st.markdown(summarize_tool_call(call, index))

    with st.expander("Show raw reasoning trace", expanded=False):
        if not run.tool_calls:
            st.write("No tool calls were made.")
        for index, call in enumerate(run.tool_calls, start=1):
            st.markdown(f"**{index}. `{call.name}`**")
            st.markdown("_Input_")
            st.code(json.dumps(call.input, indent=2), language="json")
            st.markdown("_Raw result_")
            st.code(json.dumps(call.result, indent=2), language="json")
