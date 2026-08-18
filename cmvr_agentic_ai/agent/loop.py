"""Raw Claude Messages API tool-use loop (no SDK, no LangGraph).

The agent always starts with cmvr_search, then loops ais_search until it has
enough to answer. ais_search calls are hard-capped (default 5); an overall turn
cap guards against runaway loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent import tools
from agent.llm_client import call_claude

SYSTEM_PROMPT = """\
You are a regulatory research agent for Indian automotive type-approval. You \
answer questions about which tests and requirements apply to a vehicle by \
reasoning over two document sets exposed as tools:
- cmvr_search: the Central Motor Vehicle Rules (CMVR).
- ais_search: the Automotive Industry Standards (AIS) clauses.

Follow this procedure:
1. ALWAYS call cmvr_search first for any new question. Use vehicle_category \
(e.g. 'M3' for a bus) when the query implies one.
2. Take the ais_codes returned by cmvr_search and call ais_search with them.
3. If ais_search returns further_ais_refs that you have not explored yet and \
your answer still seems incomplete, call ais_search again with those codes. \
Stop once results stop adding new relevant information, or after 2 ais_search \
calls (a hard limit).
4. When you answer, cite specific rule numbers, AIS codes, and clause headings. \
Never state a test or requirement without traceability back to a specific \
rule or clause returned by the tools.
5. If nothing relevant is found in either collection, say so explicitly. Do \
not invent rule numbers, AIS codes, or clause text.

Structure the final answer as a clear list of applicable tests/requirements, \
each with its source citation.\
"""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentRun:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    stopped_reason: str = ""


def _text_from_content(content: list[dict[str, Any]]) -> str:
    return "\n".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    ).strip()


def summarize_tool_call(call: ToolCall, index: int | None = None) -> str:
    """Return a compact, human-readable one-block summary of a tool call.

    Uses Markdown hard line breaks (two trailing spaces) so it renders cleanly
    in Streamlit / any Markdown surface.
    """
    prefix = f"{index}. " if index else ""
    args = call.input or {}
    result = call.result or {}

    if "error" in result:
        return f"{prefix}\u26a0\ufe0f **`{call.name}`** failed \u2014 {result['error']}"

    if call.name == "cmvr_search":
        query = args.get("query", "")
        category = args.get("vehicle_category")
        rules = result.get("rules", [])
        codes = result.get("ais_codes", [])
        rule_nums = ", ".join(str(r.get("ruleNumber", "?")) for r in rules) or "none"
        category_txt = f" \u00b7 category **{category}**" if category else ""
        codes_txt = ", ".join(codes) if codes else "none"
        return (
            f'{prefix}\U0001f50e **cmvr_search** \u2014 "{query}"{category_txt}  \n'
            f"\u2192 {len(rules)} rule(s): {rule_nums}  \n"
            f"\u2192 {len(codes)} AIS code(s): {codes_txt}"
        )

    if call.name == "ais_search":
        query = args.get("query", "")
        in_codes = ", ".join(args.get("ais_codes", []) or []) or "none"
        clauses = result.get("clauses", [])
        further = result.get("further_ais_refs", [])
        ids = sorted({c.get("AIS_id", "") for c in clauses if c.get("AIS_id")})
        ids_txt = ", ".join(ids) or "none"
        further_txt = ", ".join(further) if further else "none"
        return (
            f'{prefix}\U0001f4d8 **ais_search** \u2014 "{query}" \u00b7 codes: {in_codes}  \n'
            f"\u2192 {len(clauses)} clause(s) across {ids_txt}  \n"
            f"\u2192 further refs: {further_txt}"
        )

    return f"{prefix}**`{call.name}`** \u2192 {json.dumps(result)[:200]}"


def run_agent(
    user_query: str,
    *,
    max_ais_calls: int = 2,
    max_turns: int = 10,
    max_tokens: int = 4096,
    verbose: bool = True,
    on_tool_call: Callable[[ToolCall], None] | None = None,
) -> AgentRun:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]
    run = AgentRun(answer="")
    ais_calls = 0

    for _ in range(max_turns):
        run.turns += 1
        # Once the ais_search budget is spent, drop the tools so the next turn
        # forces the model to write its final answer from what it already has.
        active_tools = tools.TOOLS if ais_calls < max_ais_calls else None
        data = call_claude(
            messages, tools=active_tools, system=SYSTEM_PROMPT, max_tokens=max_tokens
        )
        content = data.get("content", [])
        messages.append({"role": "assistant", "content": content})
        stop_reason = data.get("stop_reason", "")

        if stop_reason != "tool_use":
            answer = _text_from_content(content)
            if not answer and stop_reason == "max_tokens":
                # The turn was truncated before any answer text (and may hold a
                # partial tool_use block). Discard it to keep tool_use/tool_result
                # pairing valid, then force a concise, tool-less final answer.
                messages.pop()
                retry = call_claude(
                    messages, tools=None, system=SYSTEM_PROMPT, max_tokens=max_tokens
                )
                messages.append(
                    {"role": "assistant", "content": retry.get("content", [])}
                )
                answer = _text_from_content(retry.get("content", []))
                stop_reason = retry.get("stop_reason", stop_reason)
            run.answer = answer
            run.stopped_reason = stop_reason or "end_turn"
            return run



        tool_result_blocks: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") != "tool_use":
                continue
            name = block["name"]
            tool_input = block.get("input", {}) or {}

            if name == "ais_search" and ais_calls >= max_ais_calls:
                result: dict[str, Any] = {
                    "error": f"ais_search call cap ({max_ais_calls}) reached; "
                    "answer with what you already have."
                }
            else:
                if name == "ais_search":
                    ais_calls += 1
                try:
                    result = tools.run_tool(name, tool_input)
                except Exception as error:  # surface tool failures to the model
                    result = {"error": f"{type(error).__name__}: {error}"}

            if verbose:
                print(f"\n>>> TOOL {name} input={json.dumps(tool_input)}")
                print(f"<<< RESULT {json.dumps(result)[:2000]}")

            tool_call = ToolCall(name=name, input=tool_input, result=result)
            run.tool_calls.append(tool_call)
            if on_tool_call is not None:
                on_tool_call(tool_call)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    run.answer = run.answer or "Stopped: reached the maximum number of reasoning turns."
    run.stopped_reason = "max_turns"
    return run


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Find all tests for a bus."
    result = run_agent(query)
    print("\n" + "=" * 80)
    print(f"ANSWER (stopped: {result.stopped_reason}, {len(result.tool_calls)} tool calls):\n")
    print(result.answer)
