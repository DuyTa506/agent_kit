"""Memory and RAG primitives with core Agent Kit APIs.

Run:
    python3 examples/memory_agent.py

This example loads ../.env automatically when present. It does not print any
secret values.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agent_kit import Agent
from agent_kit.memory import (
    InMemoryKeywordMemoryStore,
    MemoryContextBuilder,
    MemoryItem,
    MemorySearchTool,
    MemoryUpsertTool,
)
from agent_kit.sessions import InMemorySessionStore
from agent_kit.tools import ToolContext, ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5-nano-2025-08-07"


def load_project_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def seed_store() -> InMemoryKeywordMemoryStore:
    store = InMemoryKeywordMemoryStore()
    await store.upsert(
        [
            MemoryItem(
                id="m1",
                content="Agent Kit Scheduler V2 supports parallel read tools.",
                metadata={"label": "Scheduler note"},
                namespace="agent-kit",
            ),
            MemoryItem(
                id="m2",
                content="ToolResult can include citations and metadata for provenance.",
                metadata={"label": "Tool result note"},
                namespace="agent-kit",
            ),
            MemoryItem(
                id="m3",
                content="MemoryContextBuilder injects retrieved memory per turn only.",
                metadata={"label": "Context note"},
                namespace="agent-kit",
            ),
        ]
    )
    return store


async def local_memory_demo() -> None:
    store = await seed_store()
    search_tool = MemorySearchTool(store, namespace="agent-kit")
    ctx = ToolContext(
        cwd=str(ROOT),
        session_id="local",
        run_id="memory-demo",
        session_store=None,
    )
    result = await search_tool.execute(
        {"query": "parallel read citations", "limit": 5, "namespace": "agent-kit"},
        ctx,
    )
    print(result.summary)
    for citation in result.citations:
        print(f"  {citation.id} score={citation.score:.2f}: {citation.label}")

    builder = MemoryContextBuilder(store, namespace="agent-kit", max_tokens=80)
    hits = await store.search("How does memory context stay small?", namespace="agent-kit")
    print(f"Context builder source hits: {len(hits)}")
    print(f"Search tool resources: {search_tool.resources({'namespace': 'agent-kit'})[0]}")
    print(f"Builder type: {builder.__class__.__name__}")


async def maybe_live_agent() -> None:
    load_project_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; skipped live agent call.")
        return

    store = await seed_store()
    registry = ToolRegistry()
    registry.add(MemorySearchTool(namespace="agent-kit"))
    registry.add(MemoryUpsertTool(namespace="agent-kit"))
    agent = Agent(
        model=MODEL,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        tools=registry,
        deps=store,
        context_builder=MemoryContextBuilder(namespace="agent-kit", max_tokens=300),
        session_store=InMemorySessionStore(),
        permissions={"mode": "skip-dangerous"},
        system_prompt="Use memory context and memory tools when relevant.",
    )
    session = await agent.session()
    async for event in session.run("What do you remember about citations?"):
        if event.type == "result":
            print("Live answer:", event.final_text)


async def main() -> None:
    await local_memory_demo()
    await maybe_live_agent()


if __name__ == "__main__":
    asyncio.run(main())
