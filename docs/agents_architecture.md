# Champollion Agents — Architecture Design

## Overview

Two interactive AI agents that wrap the champollion sulcal pipeline end-to-end.
They communicate via **ACP** (Agent Communication Protocol, [i-am-bee/acp](https://github.com/i-am-bee/acp)),
use **LangGraph** for internal reasoning loops, share a **ChromaDB** vector store,
and expose a streaming interface to a GUI (Gradio → React later).

The LLM backend is configurable via an **OpenAI-compatible API** so the same
code runs against OpenAI, Anthropic (via proxy), local Ollama, or llama.cpp.

```
User (GUI)
   │  ACP Run (HTTP)
   ▼
┌──────────────────────────────┐
│  Agent 1: Pipeline Tech      │  ← triggered at pipeline start
│  LangGraph ReAct             │
│  Tools: all MCP tools        │
│         + poll_until_done    │
│         + read_preflight     │
└────────────┬─────────────────┘
             │ ACP Run (handoff on pipeline success)
             ▼
┌──────────────────────────────┐
│  Agent 2: Data Analyst       │  ← triggered when pipeline completes
│  LangGraph ReAct + RAG       │
│  Tools: ChromaDB search      │
│         + snapshot launcher  │
│         + compare launcher   │
│         + embedding reader   │
└──────────────────────────────┘
             │
             ▼
      ChromaDB (local)
      .mcp_jobs/ (job store)
```

---

## Protocol: ACP (Agent Communication Protocol)

ACP is a RESTful protocol under Linux Foundation governance ([agentcommunicationprotocol.dev](https://agentcommunicationprotocol.dev)).
Each agent is an HTTP server exposing a standard Run/Session API.
The GUI and the agents themselves communicate by creating ACP Runs.

### Why ACP here

- **Framework-agnostic**: the GUI doesn't need to know the agents are LangGraph.
- **Streaming-native**: Runs support server-sent events for real-time output.
- **Stateful sessions**: multi-turn conversation is built-in — the GUI can continue
  a session across multiple messages without re-sending history.
- **Agent-to-agent**: Agent 1 triggers Agent 2 by opening a new ACP Run against
  Agent 2's endpoint, passing the pipeline output paths as the initial message.

### ACP primitives used

| Primitive | Role |
|-----------|------|
| `Run` | Single agent execution (one pipeline invocation, one analysis request) |
| `Session` | Multi-turn GUI conversation with one agent |
| `Message` / `MessagePart` | Input/output units — text + file paths |
| `await` yield | Agent 1 pauses while a subprocess runs, resumes on status change |

### Minimal ACP agent skeleton (Python)

```python
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from collections.abc import AsyncGenerator

server = Server()

@server.agent()
async def pipeline_technician(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Orchestrates champollion pipeline stages."""
    yield {"thought": "Checking preflight..."}
    # ... LangGraph agent invocation here
    yield Message(parts=[{"content": "Pipeline complete.", "content_type": "text/plain"}])

server.run(port=8001)
```

---

## Agent 1: Pipeline Technician

### Role

Receives a pipeline request, runs `preflight_check`, adapts parameters
(GPU availability, NFS path, njobs), launches each stage via MCP tools,
suspends while subprocesses run, resumes on completion, handles failures.

### Internal architecture

```
ACP Run input (user request + paths)
    ↓
LangGraph ReAct loop
    ├─ preflight_check()       → adapt njobs, cpu flag
    ├─ start_morphologist()    → job_id
    ├─ poll_until_done()       → suspend until terminal status
    ├─ start_cortical_tiles()  → job_id
    ├─ poll_until_done()
    ├─ start_config()
    ├─ poll_until_done()
    ├─ start_embeddings()
    ├─ poll_until_done()
    ├─ start_combine()
    ├─ poll_until_done()
    └─ handoff_to_analyst()    → ACP Run on Agent 2
```

### Suspension mechanism

LangGraph checkpoints the agent state after each `poll_until_done` invocation.
The poll tool uses `asyncio.sleep` with exponential backoff rather than busy-waiting:

```python
async def poll_until_done(job_id: str, output_dir: str, interval: int = 10) -> dict:
    """Polls get_job_status until terminal; yields intermediate status to the ACP stream."""
    while True:
        state = await get_job_status(output_dir=output_dir, job_id=job_id)
        if state["status"] in ("succeeded", "failed", "cancelled"):
            return state
        await asyncio.sleep(interval)
```

The LangGraph checkpoint ensures no state is lost if the server restarts
between polls (the GUI can reconnect to the same Session).

### Parameter adaptation logic

| Condition detected by preflight | Adaptation |
|---|---|
| No CUDA device | `cpu=True` on all stages |
| `njobs` not set, >8 CPU cores | `njobs = cpu_count // 2` |
| `HF_TOKEN` absent | warn user, skip HuggingFace model download |
| Submodule missing | raise ToolError with setup instructions |

### Tools

| Tool | Source |
|---|---|
| All 17 MCP tools | `langchain-mcp-adapters` → `champollion_sulcal_mcp` |
| `poll_until_done` | local wrapper (calls `get_job_status` in a loop) |
| `read_job_log_tail` | local wrapper (calls `get_job_log`) |
| `handoff_to_analyst` | ACP client call to Agent 2 |

---

## Agent 2: Data Analyst

### Role

Receives pipeline output paths, indexes all generated data into ChromaDB,
then answers analytical questions using similarity search, outlier detection,
and on-demand visualizations via the existing pipeline scripts.

### Internal architecture

```
ACP Run input (output_dir, combined_embeddings_path)
    ↓
index_pipeline_outputs()       → populate ChromaDB collections
    ↓
LangGraph ReAct loop (interactive)
    ├─ similarity_search()     → find subjects with similar folding
    ├─ get_outliers()          → identify anomalous subjects per region
    ├─ describe_embedding()    → return 128-dim vector + UMAP coords
    ├─ run_snapshots()         → call start_snapshots MCP, return image paths
    ├─ run_compare()           → call compare.py, return diff image paths
    └─ summarize_region()      → statistical summary (mean, std, cluster count)
```

### ChromaDB collections

#### `subject_metadata`

Populated at pipeline start from the QC TSV and any demographics files found.

```
id:        "{subject_id}"
embedding: None  (metadata-only collection)
metadata:  { subject_id, qc, acquisition, session, dataset }
```

#### `region_embeddings`

Populated after `generate_embeddings` completes (one entry per subject × region).

```
id:        "{subject_id}__{region}__{hemisphere}"
embedding: float32[128]   (the champollion backbone output)
metadata:  { subject_id, region, hemisphere, dataset, run_id }
```

This is the primary collection for similarity search.

#### `pipeline_runs`

Populated by Agent 1 after each successful run — lets Agent 2 know what
was generated and where.

```
id:        "{job_id}"
embedding: None
metadata:  { stage, status, output_dir, started_at, ended_at }
```

### Indexing pipeline

```python
async def index_pipeline_outputs(combined_embeddings_path: str, output_dir: str):
    """Read all full_embeddings.csv files and upsert into ChromaDB."""
    import pandas as pd
    import chromadb

    client = chromadb.PersistentClient(path=f"{output_dir}/.chromadb")
    collection = client.get_or_create_collection("region_embeddings")

    for csv_path in Path(combined_embeddings_path).glob("**/full_embeddings.csv"):
        region, hemisphere = _parse_region_from_path(csv_path)
        df = pd.read_csv(csv_path, index_col=0)          # index = subject_id
        dim_cols = [c for c in df.columns if c.startswith("dim")]
        for subject_id, row in df.iterrows():
            collection.upsert(
                ids=[f"{subject_id}__{region}__{hemisphere}"],
                embeddings=[row[dim_cols].tolist()],
                metadatas=[{"subject_id": subject_id, "region": region, "hemisphere": hemisphere}],
            )
```

### RAG tool examples

```python
def similarity_search(query_subject: str, region: str, hemisphere: str, k: int = 10) -> list[dict]:
    """Find k subjects most similar to query_subject in the given region."""
    query_vec = collection.get(ids=[f"{query_subject}__{region}__{hemisphere}"])["embeddings"][0]
    results = collection.query(query_embeddings=[query_vec], n_results=k + 1)
    return [
        {"subject_id": m["subject_id"], "distance": d}
        for m, d in zip(results["metadatas"][0], results["distances"][0])
        if m["subject_id"] != query_subject
    ]

def get_outliers(region: str, hemisphere: str, threshold: float = 2.5) -> list[dict]:
    """Return subjects whose embedding is more than threshold std devs from the centroid."""
    all_items = collection.get(where={"region": region, "hemisphere": hemisphere})
    embeddings = np.array(all_items["embeddings"])
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    z_scores = (distances - distances.mean()) / distances.std()
    outlier_mask = z_scores > threshold
    return [
        {"subject_id": all_items["metadatas"][i]["subject_id"], "z_score": float(z_scores[i])}
        for i in np.where(outlier_mask)[0]
    ]
```

---

## LLM Backend — OpenAI-Compatible API

Both agents use `langchain_openai.ChatOpenAI` with a configurable `base_url`.
This makes the backend swappable at runtime via environment variables.

```python
from langchain_openai import ChatOpenAI

def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("CHAMPOLLION_LLM_MODEL", "gpt-4o"),
        base_url=os.environ.get("CHAMPOLLION_LLM_BASE_URL"),  # None = OpenAI default
        api_key=os.environ.get("CHAMPOLLION_LLM_API_KEY", "ollama"),
        temperature=0,
        streaming=True,
    )
```

| Backend | `CHAMPOLLION_LLM_BASE_URL` | `CHAMPOLLION_LLM_MODEL` | `CHAMPOLLION_LLM_API_KEY` |
|---|---|---|---|
| OpenAI | *(unset)* | `gpt-4o` | OpenAI key |
| Ollama | `http://localhost:11434/v1` | `llama3.1:70b` | `ollama` |
| llama.cpp | `http://localhost:8080/v1` | `local` | `none` |
| Anthropic proxy | `https://api.anthropic.com/v1` | `claude-sonnet-4-5` | Anthropic key |

---

## Handoff Protocol (Agent 1 → Agent 2)

When Agent 1 completes the `combine` stage successfully, it opens an ACP Run
on Agent 2, passing a structured handoff message:

```json
{
  "event": "pipeline_complete",
  "output_dir": "/path/to/derivatives",
  "combined_embeddings": "/path/to/derivatives/combined_embeddings",
  "run_id": "<umbrella_job_id>",
  "dataset": "UKBioBank",
  "n_subjects": 424
}
```

Agent 2 receives this as its initial `input` message, indexes the outputs,
and becomes available for interactive queries in the same ACP Session.

---

## GUI Integration

Both agents expose ACP HTTP endpoints. The GUI connects to both.

### Phase 1: Gradio (prototype)

```python
import gradio as gr
import httpx

async def chat_technician(message, history):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://localhost:8001/runs", json={...}) as r:
            async for chunk in r.aiter_text():
                yield chunk

with gr.Blocks() as demo:
    with gr.Tab("Pipeline Technician"):
        gr.ChatInterface(chat_technician, title="Pipeline Technician")
    with gr.Tab("Data Analyst"):
        gr.ChatInterface(chat_analyst, title="Data Analyst")

demo.launch()
```

### Phase 2: React (production)

React frontend with two chat panels. Agents send SSE streams; the frontend
renders image paths as `<img>` tags inline so snapshots and UMAP plots
appear directly in the conversation.

---

## Repository Layout

```
champollion_agents/           (new repo, or subfolder for now)
  agents/
    technician.py             ← ACP server + LangGraph ReAct agent
    analyst.py                ← ACP server + LangGraph + RAG tools
    handoff.py                ← shared message schema, ACP client call
  rag/
    indexer.py                ← ChromaDB population (metadata + embeddings)
    tools.py                  ← similarity_search, get_outliers, describe_embedding
  llm.py                      ← build_llm() OpenAI-compatible factory
  gui/
    app.py                    ← Gradio two-tab interface
  pixi.toml
```

### Core dependencies

```toml
[dependencies]
acp-sdk = "*"                  # ACP server + client
langchain-openai = "*"         # OpenAI-compatible LLM backend
langgraph = "*"                # ReAct loop + checkpointing
langchain-mcp-adapters = "*"   # MCP tools → LangChain tools
chromadb = "*"                 # local vector store
langchain-chroma = "*"         # LangChain ↔ ChromaDB bridge
gradio = "*"                   # Phase 1 GUI
httpx = "*"                    # ACP client calls between agents
pandas = "*"                   # embeddings CSV reading
numpy = "*"                    # outlier computation
```

---

## Open Questions

1. **Session persistence**: should ChromaDB persist across server restarts
   (default with `PersistentClient`) or be rebuilt on each run?
   Recommendation: persist, keyed by `output_dir` path.

2. **Agent 2 triggered automatically or on-demand?** Current design: Agent 1
   triggers Agent 2 automatically on pipeline success. Alternative: user
   manually opens the Analyst tab and provides `output_dir`.

3. **Security**: ACP endpoints are unauthenticated by default. For NFS/cluster
   deployments, add an API key header or bind to localhost only.

4. **Multi-run support**: the current design assumes one active pipeline run
   at a time. Supporting concurrent runs would require namespacing the
   ChromaDB collections and job store by `run_id`.
