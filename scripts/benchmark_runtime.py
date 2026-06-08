from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linch import Agent, InMemoryRunStore  # noqa: E402
from linch.sessions import InMemorySessionStore  # noqa: E402
from linch.tools import ToolContext, ToolRegistry, ToolResult  # noqa: E402
from linch.types import Usage  # noqa: E402


class DeltaProvider:
    id = "delta-bench"

    def __init__(self, *, deltas: int = 200, tool_names: list[str] | None = None) -> None:
        self.deltas = deltas
        self.tool_names = list(tool_names or [])
        self.calls = 0

    def context_window(self, model: str) -> int:
        return 100_000

    async def stream(self, req: Any) -> AsyncIterator[dict[str, object]]:
        self.calls += 1
        yield {"type": "message_start", "model": req.model}
        if self.calls == 1 and self.tool_names:
            for index, name in enumerate(self.tool_names, start=1):
                tool_id = f"call-{index}"
                yield {"type": "tool_use_start", "id": tool_id, "name": name}
                for chunk in ('{"value":"', name, '"}'):
                    yield {"type": "tool_use_input_delta", "id": tool_id, "json_delta": chunk}
                yield {"type": "tool_use_end", "id": tool_id}
            yield {"type": "message_end", "stop_reason": "tool_use", "usage": Usage()}
            return
        for _ in range(self.deltas):
            yield {"type": "text_delta", "text": "x"}
        yield {"type": "message_end", "stop_reason": "end_turn", "usage": Usage()}


class BenchTool:
    description = "Benchmark read tool."
    input_schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    scope = "read"
    parallel = True

    def __init__(self, name: str, *, delay: float = 0.005) -> None:
        self.name = name
        self.delay = delay

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def summarize(self, input: dict[str, Any]) -> str:
        return self.name

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(self.delay)
        return ToolResult(content=f"{self.name}:ok")


def registry(names: list[str]) -> ToolRegistry:
    tools = ToolRegistry()
    for name in names:
        tools.register(BenchTool(name))
    return tools


async def timed(label: str, agent: Agent, prompt: str) -> None:
    session = await agent.session()
    started = time.perf_counter()
    events = [event async for event in session.run(prompt)]
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"{label}: {elapsed_ms:.1f}ms ({len(events)} events)")


async def main() -> None:
    await timed(
        "no_tool_stream_200_deltas",
        Agent(
            model="gpt-5",
            provider=DeltaProvider(deltas=200),
            session_store=InMemorySessionStore(),
            permissions={"mode": "skip-dangerous"},
            result_offload=None,
        ),
        "stream",
    )

    tool_names = [f"Read{i}" for i in range(8)]
    await timed(
        "parallel_8_read_tools",
        Agent(
            model="gpt-5",
            provider=DeltaProvider(deltas=10, tool_names=tool_names),
            tools=registry(tool_names),
            session_store=InMemorySessionStore(),
            permissions={"mode": "skip-dangerous"},
            max_tool_concurrency=8,
            result_offload=None,
        ),
        "use tools",
    )

    await timed(
        "durable_8_read_tools",
        Agent(
            model="gpt-5",
            provider=DeltaProvider(deltas=10, tool_names=tool_names),
            tools=registry(tool_names),
            session_store=InMemorySessionStore(),
            run_store=InMemoryRunStore(),
            permissions={"mode": "skip-dangerous"},
            max_tool_concurrency=8,
            result_offload=None,
        ),
        "use durable tools",
    )


if __name__ == "__main__":
    asyncio.run(main())
