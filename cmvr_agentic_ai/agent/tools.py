"""Claude tool schemas and the dispatch bridge to the search functions."""

from __future__ import annotations

from typing import Any

from search.ais_search import ais_search
from search.cmvr_search import cmvr_search

CMVR_SEARCH_TOOL: dict[str, Any] = {
    "name": "cmvr_search",
    "description": (
        "Search the Central Motor Vehicle Rules (CMVR) for rules relevant to a "
        "query. Always call this FIRST for a new question. Returns matched rules "
        "(rule number, title, text snippet) and the union of AIS standard codes "
        "those rules cross-reference. Use those AIS codes to drive ais_search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what to find in the CMVR.",
            },
            "vehicle_category": {
                "type": "string",
                "description": (
                    "Optional vehicle category code to bias results, e.g. 'M3' "
                    "(large passenger vehicle / bus), 'N2', 'L7'. Omit if unknown."
                ),
            },
        },
        "required": ["query"],
    },
}

AIS_SEARCH_TOOL: dict[str, Any] = {
    "name": "ais_search",
    "description": (
        "Search the Automotive Industry Standards (AIS) clauses. Call this after "
        "cmvr_search, passing the ais_codes it returned. The ais_codes act as a "
        "HARD pre-filter: only clauses whose own AIS_id is in the list are "
        "returned, ranked by relevance to the query. Returns matching clauses "
        "(AIS_id, heading, subheading, description snippet) and 'further_ais_refs': "
        "additional AIS standards those clauses reference. If further_ais_refs "
        "contains unexplored codes and the answer is still incomplete, call "
        "ais_search again with them. Stop when results stop adding new relevant "
        "information (hard cap: 2 ais_search calls)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for in the AIS clauses (e.g. the test/requirement topic).",
            },
            "ais_codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "AIS standard codes to restrict the search to, e.g. "
                    "['AIS-052', 'AIS-090']. Used as a HARD pre-filter on each "
                    "clause's own AIS_id \u2014 only clauses belonging to these "
                    "standards are returned. Pass the codes from cmvr_search or "
                    "the further_ais_refs of a prior ais_search."
                ),
            },
        },
        "required": ["query", "ais_codes"],
    },
}

TOOLS: list[dict[str, Any]] = [CMVR_SEARCH_TOOL, AIS_SEARCH_TOOL]

_DISPATCH = {
    "cmvr_search": lambda args: cmvr_search(
        query=args["query"], vehicle_category=args.get("vehicle_category")
    ),
    "ais_search": lambda args: ais_search(
        query=args["query"], ais_codes=args.get("ais_codes") or []
    ),
}


def run_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name; raises KeyError for unknown tools."""
    if name not in _DISPATCH:
        raise KeyError(f"Unknown tool: {name}")
    return _DISPATCH[name](tool_input)
