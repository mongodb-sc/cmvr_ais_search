# Building an Agentic Regulatory Research Assistant for CMVR and AIS

## Introduction

What if a user could ask, *“What tests apply to an M3 bus?”* and get back a clean, cited answer grounded in real regulatory sources instead of a vague LLM summary?

That is exactly what this project does.

The codebase in [cmvr_agentic_ai/](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai) builds a **regulatory research assistant** for Indian automotive type approval. It searches two knowledge sources:

- **CMVR**: Central Motor Vehicle Rules
- **AIS**: Automotive Industry Standards

The interesting part is that this is not just “chat with a database.” The system uses an **agentic loop** that follows a disciplined workflow:

1. Search **CMVR** first
2. Extract referenced **AIS codes**
3. Search **AIS** using those codes as a strict filter
4. Return a **traceable answer with citations**

That design makes the system much more useful for compliance-style workflows, where being correct is not enough—you also need to show **where the answer came from**.

## Prerequisites

To follow this project comfortably, you should know:

- **Python**
- **FastAPI**
- **React / Next.js**
- Basic **MongoDB** concepts
- The idea of **vector embeddings** and **hybrid search**
- How LLM **tool calling** works

## High-Level Overview

At a high level, the application is split into four layers:

1. **Configuration and infrastructure**  
   Shared environment variables, MongoDB connections, and model settings live in [config.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/config.py).

2. **Search layer**  
   The retrieval pipeline in [search/](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search) combines:
   - MongoDB Atlas vector search
   - Atlas full-text search
   - Voyage AI reranking

3. **Agent layer**  
   The orchestration code in [agent/](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/agent) defines the tools, calls the LLM, and enforces the reasoning sequence.

4. **Delivery layer**  
   Results are exposed through:
   - a streaming API in [api.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/api.py)
   - a Next.js frontend in [web/src/components/research-workspace.tsx](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/web/src/components/research-workspace.tsx)
   - optional Gradio and Streamlit interfaces in [gradio_app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/gradio_app.py) and [app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/app.py)

## Module-by-Module Walkthrough

## 1. Central configuration and shared handles

Everything starts in [config.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/config.py).

```python
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "automotive_regulations")
CMVR_COLLECTION = os.getenv("CMVR_RULE_COLLECTION", "cmvr_rules")
AIS_COLLECTION = os.getenv("AIS_RULE_COLLECTION", "ais_rules")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-large")
RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
```

This file does two important jobs:

- It defines **all infrastructure-level settings** in one place.
- It exposes **lazy MongoDB helpers** so the rest of the code does not need to know connection details.

```python
@lru_cache(maxsize=1)
def mongo_client() -> Any:
    from pymongo import MongoClient
    client = MongoClient(
        MONGODB_URI,
        appname="cmvr-agentic-ai",
        serverSelectionTimeoutMS=10_000,
        tz_aware=True,
    )
    client.admin.command("ping")
    return client
```

### Why this design works

The `@lru_cache` choice is subtle but smart. It ensures the app creates **one process-wide Mongo client**, which is efficient and avoids reconnecting on every request.

The helper functions below keep calling code clean:

```python
def cmvr_collection() -> Any:
    return mongo_client()[MONGODB_DATABASE][CMVR_COLLECTION]

def ais_collection() -> Any:
    return mongo_client()[MONGODB_DATABASE][AIS_COLLECTION]
```

That means retrieval modules can focus on **search logic**, not plumbing.

## 2. Embeddings and reranking: the retrieval foundation

The files [search/embeddings.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search/embeddings.py) and [search/rerank.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search/rerank.py) wrap Voyage AI.

### Query embeddings

```python
def embed_query(text: str) -> list[float]:
    response = _client().embed(
        [text],
        model=config.EMBEDDING_MODEL,
        input_type="query",
        truncation=True,
        output_dtype="float",
        output_dimension=config.EMBEDDING_DIMENSION,
    )
    return [float(value) for value in response.embeddings[0]]
```

### Reranking

```python
def rerank(query: str, documents: list[str], *, top_k: int) -> list[RerankHit]:
    response = _client().rerank(
        query,
        list(documents),
        model=config.RERANK_MODEL,
        top_k=min(top_k, len(documents)),
        truncation=True,
    )
    return [
        RerankHit(index=int(result.index), relevance_score=float(result.relevance_score))
        for result in response.results
    ]
```

### Why split embedding and reranking?

Because they solve different problems:

- **Embeddings** help find semantically similar documents.
- **Reranking** improves precision after candidate retrieval.

This is a common production pattern: use fast retrieval to get a shortlist, then use a stronger ranking model to sort that shortlist more intelligently.

Both wrappers create the Voyage client lazily, which keeps imports safe for offline or partial workflows.

## 3. Hybrid search with MongoDB Atlas `$rankFusion`

The heart of the retrieval system lives in [search/hybrid.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search/hybrid.py).

The project does not rely on only vector search or only keyword search. Instead, it **fuses both**.

```python
def rank_fusion(
    collection: Any,
    *,
    vector_pipeline: list[dict[str, Any]],
    text_pipeline: list[dict[str, Any]],
    limit: int,
    projection: dict[str, Any] | None,
    vector_weight: float = config.VECTOR_WEIGHT,
    text_weight: float = config.LEXICAL_WEIGHT,
) -> list[dict[str, Any]]:
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": vector_pipeline,
                        "lexical": text_pipeline,
                    }
                },
                "combination": {
                    "weights": {"vector": vector_weight, "lexical": text_weight}
                },
            }
        },
        {"$limit": limit},
    ]
    if projection:
        pipeline.append({"$project": projection})
    return list(collection.aggregate(pipeline))
```

### Why this matters

Regulatory text is tricky:

- Sometimes users use the **exact clause wording**.
- Sometimes they ask in **plain natural language**.

A pure lexical search misses semantic phrasing. A pure vector search can miss exact identifiers. By combining both, the system gets the best of each approach.

Then the module adds a second quality layer:

```python
def hybrid_rerank(...):
    candidates = rank_fusion(...)
    if not candidates:
        return []

    documents = [rerank_text(doc)[: config.MAX_RERANK_DOCUMENT_CHARS] for doc in candidates]
    hits = rerank.rerank(query, documents, top_k=rerank_top_k)
```

This keeps the search pipeline reusable. Both CMVR and AIS searches can use the same retrieval engine with different projections and formatting logic.

## 4. CMVR search: find governing rules first

The first domain-specific tool lives in [search/cmvr_search.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search/cmvr_search.py).

```python
def cmvr_search(
    query: str,
    vehicle_category: str | None = None,
    *,
    top_k: int = config.CMVR_TOP_K,
    candidate_limit: int = config.CANDIDATE_LIMIT,
) -> dict[str, Any]:
```

The function slightly enriches the query when a vehicle category is known:

```python
effective_query = (
    f"{query} (vehicle category {vehicle_category})" if vehicle_category else query
)
query_vector = embeddings.embed_query(effective_query)
```

It also builds a weighted lexical query:

```python
should = [
    {"text": {"query": effective_query, "path": "canonicalTitle", "score": {"boost": {"value": 3}}}},
    {"text": {"query": effective_query, "path": "ruleText"}},
]
if vehicle_category:
    should.append(
        {"text": {"query": vehicle_category, "path": "ruleText", "score": {"boost": {"value": 2}}}}
    )
```

### Why this is thoughtful

Notice that `vehicle_category` is used to **bias** results, not hard-filter them. That is a good fit for regulation search. Hard filtering could hide relevant rules that discuss the topic broadly without repeating the category code in a predictable way.

After retrieval, the function extracts AIS references:

```python
for doc in docs:
    codes = _ais_codes(doc)
    all_codes.update(codes)
    rules.append(
        {
            "ruleNumber": doc.get("ruleNumber", ""),
            "canonicalTitle": doc.get("canonicalTitle", ""),
            "ruleText_snippet": snippet(doc.get("ruleText", "")),
            "ais_codes": codes,
            "rerank_score": round(doc["_rerank_score"], 4),
        }
    )
```

This is the bridge between the two knowledge bases. CMVR is not just an answer source—it is also the mechanism that tells the agent **which AIS standards are worth searching next**.

## 5. AIS search: constrained, traceable clause retrieval

The second domain tool is [search/ais_search.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search/ais_search.py).

Its most important behavior is this:

```python
if codes:
    compound["filter"] = [{"in": {"path": "AIS_id", "value": codes}}]
    vector_filter = {"AIS_id": {"$in": codes}}
```

### Why this is the right choice

This is not a loose hint. It is a **hard pre-filter**.

That means once CMVR says “these AIS codes are relevant,” the AIS search only looks inside those standards. This makes the system:

- more relevant
- easier to audit
- less likely to hallucinate unrelated standards

The result structure is also designed for iterative research:

```python
for doc in docs:
    clauses.append(
        {
            "AIS_id": doc.get("AIS_id", ""),
            "rule": doc.get("rule", ""),
            "heading": doc.get("heading", ""),
            "subheading": doc.get("subheading"),
            "description_snippet": snippet(doc.get("description", "")),
            "rerank_score": round(doc["_rerank_score"], 4),
        }
    )
    for ref in doc.get("AIS") or []:
        if ref and ref not in seen:
            further.add(ref)
```

The returned `further_ais_refs` gives the agent a way to expand only when needed.

This is a nice example of **bounded agentic behavior**: the model can loop, but only through structured evidence trails.

## 6. Tool schemas and dispatch

The LLM does not call Python functions directly. It uses declared tool schemas from [agent/tools.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/agent/tools.py).

```python
CMVR_SEARCH_TOOL = {
    "name": "cmvr_search",
    "description": (
        "Search the Central Motor Vehicle Rules (CMVR) for rules relevant to a "
        "query. Always call this FIRST for a new question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "vehicle_category": {"type": "string"},
        },
        "required": ["query"],
    },
}
```

And similarly for AIS:

```python
AIS_SEARCH_TOOL = {
    "name": "ais_search",
    "description": (
        "Search the Automotive Industry Standards (AIS) clauses. Call this after "
        "cmvr_search, passing the ais_codes it returned."
    ),
    ...
}
```

Execution is then routed through a tiny dispatcher:

```python
_DISPATCH = {
    "cmvr_search": lambda args: cmvr_search(
        query=args["query"], vehicle_category=args.get("vehicle_category")
    ),
    "ais_search": lambda args: ais_search(
        query=args["query"], ais_codes=args.get("ais_codes") or []
    ),
}
```

### Why this module matters

This file is where search functions become **LLM-callable tools**.

The descriptions are especially important. They do not just document behavior for developers—they also teach the model the intended workflow.

That is a core pattern in tool-using systems: part of your control logic lives in code, and part lives in **carefully written tool instructions**.

## 7. The raw agent loop

The orchestration logic lives in [agent/loop.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/agent/loop.py), and it is one of the strongest parts of the codebase.

The system prompt is explicit:

```python
SYSTEM_PROMPT = """\
You are a regulatory research agent for Indian automotive type-approval.
...
1. ALWAYS call cmvr_search first for any new question.
2. Take the ais_codes returned by cmvr_search and call ais_search with them.
3. ... Stop once results stop adding new relevant information, or after 2 ais_search calls.
4. When you answer, cite specific rule numbers, AIS codes, and clause headings.
"""
```

Then the loop enforces those constraints at runtime:

```python
for _ in range(max_turns):
    active_tools = tools.TOOLS if ais_calls < max_ais_calls else None
    data = call_claude(
        messages, tools=active_tools, system=SYSTEM_PROMPT, max_tokens=max_tokens
    )
```

### Why disabling tools is clever

Once the AIS call budget is exhausted, `active_tools` becomes `None`. That forces the model to stop searching and produce a final answer from the evidence it already has.

This is a simple but effective guardrail.

Tool calls are executed and fed back into the conversation:

```python
for block in content:
    if block.get("type") != "tool_use":
        continue

    name = block["name"]
    tool_input = block.get("input", {}) or {}
    result = tools.run_tool(name, tool_input)

    tool_result_blocks.append(
        {
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": json.dumps(result),
        }
    )

messages.append({"role": "user", "content": tool_result_blocks})
```

### Why this implementation is good

This loop stays close to the underlying Messages API instead of hiding everything behind a framework. That makes the behavior easier to understand:

- user sends a prompt
- model asks for a tool
- Python runs the tool
- tool output is returned
- model continues reasoning

It is “agentic,” but still transparent.

## 8. Low-level LLM gateway integration

The only module that actually talks to the model is [agent/llm_client.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/agent/llm_client.py).

```python
def call_claude(
    messages: list[dict],
    tools: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int = 2000,
) -> dict:
    api_key = os.environ.get("GROVE_API_KEY")
    if not api_key:
        raise RuntimeError("GROVE_API_KEY is not set")
```

The request is sent via `httpx`:

```python
response = httpx.post(
    config.GROVE_BASE_URL,
    headers={
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "api-key": api_key,
    },
    json=payload,
    timeout=60,
)
```

### Why keep this thin?

This module deliberately does almost nothing except adapt request format.

That separation is valuable because it keeps:

- network concerns in one file
- orchestration concerns in [agent/loop.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/agent/loop.py)
- retrieval concerns in [search/](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/search)

When systems grow, this kind of boundary pays off quickly.

## 9. Streaming API and short-term history

The HTTP layer in [api.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/api.py) turns the agent into a usable backend.

The key feature is streaming:

```python
@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request.query.strip()),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Instead of waiting for one big response, the backend sends **newline-delimited JSON events** as the run progresses.

Inside `_event_stream`, tool calls are published before the final answer:

```python
def on_tool_call(call: Any) -> None:
    payload = {
        "index": step,
        "name": call.name,
        "input": call.input,
        "result": call.result,
        "summary": summarize_tool_call(call, step),
    }
    events.put(("tool", payload))
```

That means the frontend can show a live research trace, not just a spinner.

### History retention

The API also persists a small recent history:

```python
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
```

Then it trims the collection to the newest five items:

```python
stale_ids = [
    document["_id"]
    for document in collection.find({}, {"_id": 1})
    .sort("createdAt", -1)
    .skip(5)
]
if stale_ids:
    collection.delete_many({"_id": {"$in": stale_ids}})
```

This is a pragmatic choice: enough persistence for usability, without turning the demo into a full audit database.

## 10. The Next.js research workspace

The main frontend experience lives in [web/src/components/research-workspace.tsx](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/web/src/components/research-workspace.tsx), rendered through [web/src/app/page.tsx](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/web/src/app/page.tsx).

The component starts a streaming request here:

```tsx
const response = await fetch(`${API_URL}/api/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: cleanQuery }),
});
```

Then it reads NDJSON chunks manually:

```tsx
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  buffer += decoder.decode(value, { stream: !done });
  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";

  for (const line of lines) {
    if (!line.trim()) continue;
    const streamEvent = JSON.parse(line) as StreamEvent;
    if (streamEvent.type === "tool") {
      setSteps((current) => [...current, streamEvent]);
    } else if (streamEvent.type === "done") {
      setAnswer(streamEvent.answer || "No answer produced.");
      setMeta(streamEvent);
    }
  }
}
```

### Why this is a strong UX choice

This interface does not pretend the model magically “knows” the answer. It shows the path:

- which tool ran
- what it found
- how many clauses or rules were returned

That improves trust, especially for compliance or standards research.

The component also supports recent history:

```tsx
const response = await fetch(`${API_URL}/api/history/${entry.id}`);
const detail = await response.json() as HistoryDetail;
setSteps(detail.steps);
setAnswer(detail.answer);
```

So users can revisit earlier investigations without rerunning the full agent.

## 11. Styling and application shell

The app shell is defined in [web/src/app/layout.tsx](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/web/src/app/layout.tsx), which sets up metadata and typography.

```tsx
export const metadata: Metadata = {
  title: "CMVR / AIS Regulatory Intelligence",
  description: "Agentic automotive type-approval research with cited CMVR and AIS evidence.",
};
```

The visual styling in [web/src/components/research-workspace.module.css](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/web/src/components/research-workspace.module.css) creates a dashboard-like workspace with:

- a query panel
- a results panel
- a collapsible evidence trace
- responsive layouts for smaller screens

The styling is not just decorative. It reinforces the product model: this is a **research workspace**, not a generic chatbot.

## 12. Alternate interfaces: Gradio and Streamlit

The project also includes lightweight Python-first frontends:

- [gradio_app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/gradio_app.py)
- [app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/app.py)

In [gradio_app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/gradio_app.py), tool calls are streamed into the chat UI as reasoning steps:

```python
def on_tool_call(call) -> None:
    events.put(("tool", call))

run = run_agent(user_message, on_tool_call=on_tool_call, verbose=True)
```

And in [app.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/app.py), Streamlit shows a status-driven workflow:

```python
with st.status("Reasoning over CMVR and AIS collections…", expanded=True) as status:
    def on_tool_call(call) -> None:
        step["n"] += 1
        status.update(label=f"Step {step['n']}: {call.name}…")
        status.markdown(summarize_tool_call(call, step["n"]))
```

### Why keep multiple interfaces?

Because they serve different needs:

- **Next.js UI** for a polished end-user experience
- **Gradio** for quick demos
- **Streamlit** for lightweight exploration and debugging

All three reuse the same backend logic, which is a good sign that the core architecture is well separated.

## 13. Operational scripts: indexes and embeddings

Two utility scripts make the retrieval layer production-ready:

- [db/indexes.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/db/indexes.py)
- [db/embed_ais.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/db/embed_ais.py)

### Creating search indexes

```python
SearchIndexModel(
    name=config.AIS_TEXT_VECTOR_INDEX,
    type="vectorSearch",
    definition={
        "fields": [
            {
                "type": "vector",
                "path": config.AIS_TEXT_VECTOR_FIELD,
                "numDimensions": config.EMBEDDING_DIMENSION,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "AIS_id"},
        ]
    },
)
```

### Generating embeddings in batches

```python
def _token_batches(texts: list[str]) -> Iterator[tuple[list[int], list[str]]]:
    token_counts = embeddings.count_tokens(texts)
    ...
    if batch and batch_tokens + capped > config.EMBEDDING_REQUEST_TOKEN_LIMIT:
        yield indices, batch
```

### Why these scripts matter

A good agent depends on good retrieval, and good retrieval depends on **proper indexing and embedding hygiene**.

The embedding script is especially careful about token budgets, which is exactly the kind of operational detail that often gets skipped in prototypes.

## 14. Tests that protect the API contract

The existing test coverage in [tests/test_api.py](/Users/utsav.talwar/Desktop/UST/ust-demo/cmvr_agentic_ai/tests/test_api.py) focuses on the API’s most important promises.

For example, this test verifies that tool events are emitted before the final answer:

```python
events = [json.loads(line) for line in _event_stream("bus tests")]
self.assertEqual([event["type"] for event in events], ["tool", "done"])
```

And this one checks history trimming:

```python
collection.find.return_value.sort.return_value.skip.assert_called_once_with(5)
collection.delete_many.assert_called_once_with(
    {"_id": {"$in": ["old-1", "old-2"]}}
)
```

### Why these tests are valuable

They are not testing implementation trivia. They are testing the **behavioral contract** the frontend depends on:

- stream ordering
- persistence behavior
- bounded history retention

That is exactly the right level for this kind of app.

## Conclusion

This project is a strong example of how to build an **agentic application without overengineering it**.

It combines:

- a clear **retrieval strategy**
- a disciplined **tool-calling loop**
- a practical **streaming API**
- multiple usable **frontends**
- and just enough **operational tooling** to support real data workflows

What makes the architecture compelling is not just that it uses an LLM. It is that the LLM is placed inside a system with **rules, boundaries, and evidence flow**:

- CMVR always comes first
- AIS search is constrained by CMVR references
- answers must be cited
- tool usage is bounded

That makes the assistant much more trustworthy than a generic chat experience.

### Possible next steps

If you wanted to extend this project, good next features would be:

- **conversation memory** across multiple research questions
- **stronger citation formatting** with deep links to clauses
- **filters by vehicle category or subsystem**
- **exportable reports** for homologation teams
- **auth and user-specific saved workspaces**
- **more tests** around search tool behavior and failure paths

If you are building any kind of evidence-first AI system—especially in legal, regulatory, or standards-heavy domains—this codebase is a very solid pattern to study.
