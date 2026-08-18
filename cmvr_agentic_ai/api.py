"""HTTP streaming adapter for the CMVR/AIS agent.

Run with: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from agent.loop import run_agent, summarize_tool_call


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)


app = FastAPI(title="CMVR/AIS Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


def _serialize_history(document: dict[str, Any], *, include_output: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(document["_id"]),
        "query": document["query"],
        "createdAt": document["createdAt"].isoformat(),
        "toolCalls": document.get("toolCalls", 0),
    }
    if include_output:
        payload.update(
            {
                "answer": document.get("answer", ""),
                "steps": document.get("steps", []),
                "turns": document.get("turns", 0),
                "stoppedReason": document.get("stoppedReason", ""),
            }
        )
    return payload


def _save_history(query: str, result: Any, steps: list[dict[str, Any]]) -> None:
    collection = config.history_collection()
    collection.insert_one(
        {
            "query": query,
            "answer": result.answer,
            "steps": steps,
            "turns": result.turns,
            "toolCalls": len(result.tool_calls),
            "stoppedReason": result.stopped_reason,
            "createdAt": datetime.now(timezone.utc),
        }
    )
    stale_ids = [
        document["_id"]
        for document in collection.find({}, {"_id": 1})
        .sort("createdAt", -1)
        .skip(5)
    ]
    if stale_ids:
        collection.delete_many({"_id": {"$in": stale_ids}})


def _event_stream(query: str) -> Iterator[str]:
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    step = 0
    history_steps: list[dict[str, Any]] = []

    def on_tool_call(call: Any) -> None:
        nonlocal step
        step += 1
        payload = {
            "index": step,
            "name": call.name,
            "input": call.input,
            "result": call.result,
            "summary": summarize_tool_call(call, step),
        }
        history_steps.append(payload)
        events.put(("tool", payload))

    def worker() -> None:
        try:
            result = run_agent(query, on_tool_call=on_tool_call, verbose=False)
            try:
                _save_history(query, result, history_steps)
            except Exception:
                pass
            events.put(
                (
                    "done",
                    {
                        "answer": result.answer,
                        "turns": result.turns,
                        "toolCalls": len(result.tool_calls),
                        "stoppedReason": result.stopped_reason,
                    },
                )
            )
        except Exception as error:
            events.put(
                (
                    "error",
                    {"message": f"{type(error).__name__}: {error}"},
                )
            )

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = events.get()
        yield json.dumps({"type": kind, **payload}) + "\n"
        if kind in {"done", "error"}:
            return


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history")
def list_history() -> list[dict[str, Any]]:
    documents = config.history_collection().find({}).sort("createdAt", -1).limit(5)
    return [_serialize_history(document, include_output=False) for document in documents]


@app.get("/api/history/{history_id}")
def get_history(history_id: str) -> dict[str, Any]:
    if not ObjectId.is_valid(history_id):
        raise HTTPException(status_code=404, detail="History entry not found")
    document = config.history_collection().find_one({"_id": ObjectId(history_id)})
    if document is None:
        raise HTTPException(status_code=404, detail="History entry not found")
    return _serialize_history(document, include_output=True)


@app.delete("/api/history", status_code=204)
def clear_history() -> Response:
    config.history_collection().delete_many({})
    return Response(status_code=204)


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request.query.strip()),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
