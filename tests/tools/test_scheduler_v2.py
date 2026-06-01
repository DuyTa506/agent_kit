from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_kit import Agent
from agent_kit.abort import AbortContext
from agent_kit.events import ToolCallEndEvent
from agent_kit.permissions import PermissionEngine
from agent_kit.providers import BaseProvider
from agent_kit.scheduler import execute_tool_calls
from agent_kit.tools import ResourceAccess, ToolContext, ToolRegistry, ToolResult, ToolScope
from agent_kit.types import ToolUseBlock


class Recorder:
    def __init__(self) -> None:
        self.running = 0
        self.max_running = 0
        self.timeline: list[tuple[str, str, float, int]] = []

    def start(self, name: str) -> None:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.timeline.append(("start", name, time.perf_counter(), self.running))

    def end(self, name: str) -> None:
        self.timeline.append(("end", name, time.perf_counter(), self.running))
        self.running -= 1

    def started_before_end(self, later: str, earlier: str) -> bool:
        later_start = next(
            t for kind, name, t, _ in self.timeline if kind == "start" and name == later
        )
        earlier_end = next(
            t for kind, name, t, _ in self.timeline if kind == "end" and name == earlier
        )
        return later_start < earlier_end


class TimedTool:
    description = "Timed test tool."
    input_schema = {"type": "object", "properties": {"resource": {"type": "string"}}}

    def __init__(
        self,
        name: str,
        recorder: Recorder,
        *,
        scope: ToolScope = "read",
        parallel: bool = True,
        delay: float = 0.02,
        output: str | None = None,
    ) -> None:
        self.name = name
        self.recorder = recorder
        self.scope: ToolScope = scope
        self.parallel = parallel
        self.parallel_safe = parallel
        self.delay = delay
        self.output = output or name

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def summarize(self, input: dict[str, Any]) -> str:
        return self.name

    def resources(self, input: dict[str, Any]) -> list[ResourceAccess]:
        resource = input.get("resource")
        if not isinstance(resource, str):
            return []
        mode = "read" if self.scope == "read" else "write"
        return [ResourceAccess(resource=resource, mode=mode)]

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.recorder.start(self.name)
        try:
            await asyncio.sleep(self.delay)
            return ToolResult(content=self.output, summary=self.name)
        finally:
            self.recorder.end(self.name)


class DummyProvider(BaseProvider):
    async def stream(self, request):  # pragma: no cover - not used by this test module
        if False:
            yield {}

    def context_window(self, model: str) -> int:
        return 1000


def make_agent(registry: ToolRegistry, *, max_tool_concurrency: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        cwd=".",
        tools=registry,
        permission_engine=PermissionEngine(mode="skip-dangerous"),
        max_tool_concurrency=max_tool_concurrency,
        tool_concurrency=max_tool_concurrency,
    )


def make_session() -> SimpleNamespace:
    return SimpleNamespace(
        id="s1",
        store=None,
        active_run_id="run-1",
        tools_override=None,
        current_turn_allowed_tools=None,
    )


async def run_calls(registry: ToolRegistry, blocks: list[ToolUseBlock], *, limit: int = 8):
    events = [
        event
        async for event in execute_tool_calls(
            blocks,
            make_agent(registry, max_tool_concurrency=limit),
            make_session(),
            AbortContext(),
        )
    ]
    return events


@pytest.mark.asyncio
async def test_read_tools_run_in_parallel() -> None:
    recorder = Recorder()
    registry = ToolRegistry()
    registry.add(TimedTool("ReadA", recorder, scope="read", parallel=True))
    registry.add(TimedTool("ReadB", recorder, scope="read", parallel=True))

    await run_calls(
        registry,
        [
            ToolUseBlock(id="a", name="ReadA", input={"resource": "file:a"}),
            ToolUseBlock(id="b", name="ReadB", input={"resource": "file:b"}),
        ],
    )

    assert recorder.max_running == 2
    assert recorder.started_before_end("ReadB", "ReadA")


@pytest.mark.asyncio
async def test_write_tools_serialize_even_when_marked_parallel() -> None:
    recorder = Recorder()
    registry = ToolRegistry()
    registry.add(TimedTool("WriteA", recorder, scope="write", parallel=True))
    registry.add(TimedTool("WriteB", recorder, scope="write", parallel=True))

    await run_calls(
        registry,
        [
            ToolUseBlock(id="a", name="WriteA", input={"resource": "file:a"}),
            ToolUseBlock(id="b", name="WriteB", input={"resource": "file:b"}),
        ],
    )

    assert recorder.max_running == 1


@pytest.mark.asyncio
async def test_read_write_same_resource_do_not_overlap() -> None:
    recorder = Recorder()
    registry = ToolRegistry()
    registry.add(TimedTool("ReadA", recorder, scope="read", parallel=True))
    registry.add(TimedTool("WriteA", recorder, scope="write", parallel=True))

    await run_calls(
        registry,
        [
            ToolUseBlock(id="read", name="ReadA", input={"resource": "file:a"}),
            ToolUseBlock(id="write", name="WriteA", input={"resource": "file:a"}),
        ],
    )

    assert recorder.max_running == 1


@pytest.mark.asyncio
async def test_max_tool_concurrency_is_enforced() -> None:
    recorder = Recorder()
    registry = ToolRegistry()
    for name in ("ReadA", "ReadB", "ReadC"):
        registry.add(TimedTool(name, recorder, scope="read", parallel=True))

    await run_calls(
        registry,
        [
            ToolUseBlock(id="a", name="ReadA", input={"resource": "file:a"}),
            ToolUseBlock(id="b", name="ReadB", input={"resource": "file:b"}),
            ToolUseBlock(id="c", name="ReadC", input={"resource": "file:c"}),
        ],
        limit=2,
    )

    assert recorder.max_running == 2


def test_agent_accepts_max_tool_concurrency_option() -> None:
    agent = Agent(model="dummy", provider=DummyProvider(), max_tool_concurrency=3)

    assert agent.max_tool_concurrency == 3
    assert agent.tool_concurrency == 3


@pytest.mark.asyncio
async def test_results_keep_original_tool_call_order() -> None:
    recorder = Recorder()
    registry = ToolRegistry()
    registry.add(TimedTool("Slow", recorder, delay=0.03, output="slow"))
    registry.add(TimedTool("Fast", recorder, delay=0.005, output="fast"))

    events = await run_calls(
        registry,
        [
            ToolUseBlock(id="slow", name="Slow", input={"resource": "file:slow"}),
            ToolUseBlock(id="fast", name="Fast", input={"resource": "file:fast"}),
        ],
    )

    end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
    assert [e.tool_use_id for e in end_events] == ["slow", "fast"]
    assert [e.result for e in end_events] == ["slow", "fast"]
