# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental one-child agent; inline mode uses the same tools and loop."""

import asyncio
import json
import sys
import time
import uuid

import httpx
from harbor.agents.base import BaseAgent

from .delegation import decide_delegation, DelegationEstimate
from .streaming import ChatCompletionStream

SYSTEM = """Solve the user's task in the container. Inspect before editing, use tools for evidence,
verify the result, then give a concise final answer. Do not inspect benchmark verifier files or /tests.
The parent owns edits and final verification. A delegated child investigates only its assigned
question without modifying files or delegating. Do not edit files a running child is inspecting.

Optimize time to verified correct completion, including all parent and child work.
Batching can share weight reads, but each sequence adds computation and attention traffic.
Longer attended context costs more per step; sliding-window attention can cap this cost.
Available decode slots do not imply free capacity.
Prefill taxes shared decode. Prefix caching avoids repeated prefill, not later attention cost.
Shorter context does not necessarily free allocated memory.

The harness can decline a spawn. If declined, continue inline without retrying delegation.
Do not infer permission from idle hardware or calculate performance estimates yourself.

Before spawning: choose one necessary independent question and an unfinished parent shell action.
Pass that action as parent_command; the harness runs it alongside the child and returns its output.
Delegate before solving the child's question. If nothing independent remains, work inline.
Use at most the available child budget.
Do not duplicate work.
Do not spawn children merely to occupy idle hardware.
Each child assignment must contain:
- One concrete question or deliverable.
- Necessary context and source paths; the child does not inherit your conversation.
- A bounded output budget and a request for concise, supported findings.
Children may use shell for read-only inspection and computation. Return findings directly; no extra
summary call is needed. Child results arrive between turns. Wait only when the result is needed;
cancel obsolete work. Incorporate required findings before finishing.
Faster failure, unnecessary reasoning, and higher token throughput are not improvements.
Do not narrate performance arithmetic; act on the task.
"""


def function(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    function(
        "shell",
        "Run a shell command in the container.",
        {"command": {"type": "string"}},
    ),
    function(
        "spawn",
        "Start a bounded child and run an independent parent shell command; return the command's output.",
        {
            "question": {
                "type": "string",
                "description": "Self-contained question, source paths and necessary evidence; at most 4096 UTF-8 bytes.",
            },
            "parent_command": {
                "type": "string",
                "description": "An unfinished, necessary shell action independent of the child's question.",
            },
        },
    ),
    function("wait", "Wait for the child's findings if they are now needed.", {}),
    function("cancel", "Cancel an obsolete child and release its session.", {}),
]


class ResourceAgent(BaseAgent):
    def __init__(
        self,
        *args,
        base_url,
        enabled=False,
        max_tokens=4096,
        child_output_tokens=1536,
        step_limit=100,
        temperature=0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (
            type(enabled) is not bool
            or min(max_tokens, child_output_tokens, step_limit) < 1
        ):
            raise ValueError("Expected a boolean mode and positive budgets")
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.child_output_tokens = child_output_tokens
        self.child_budget = child_output_tokens
        self.step_limit = step_limit
        self.temperature = temperature
        self.child = None
        self.child_messages = []
        self.child_delivered = False
        self.child_used = False
        self.sessions = set()
        self.records = []
        self.output_tokens = {"parent": 0, "child": 0}
        self.engine_samples = {}
        self.started = time.monotonic()
        self.prefix = str(uuid.uuid4())
        self.client = None
        self.context = None

    @staticmethod
    def name():
        return "resource-poc"

    def version(self):
        return "0.1"

    async def setup(self, environment):
        pass

    def record(self, kind, **data):
        event = {"type": kind, "at_s": time.monotonic() - self.started, **data}
        self.records.append(event)
        with (self.logs_dir / "events.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")

    async def report(self, turn):
        response = await self.client.get(self.base_url + "/runtime/stats")
        response.raise_for_status()
        snapshot = response.json()
        self.record("request_stats", turn=turn, snapshot=snapshot)
        samples = snapshot["recent_generations"]
        for sample in samples:
            if sample["session_id"] in {
                self.prefix + "-parent",
                self.prefix + "-child",
            }:
                self.engine_samples[sample["id"]] = sample
        return snapshot

    def estimate_delegation(self, messages, snapshot) -> DelegationEstimate | None:
        """Override with calibrated workload costs; server rates alone are insufficient."""
        return None

    def admission(self, messages, snapshot, phase):
        available = self.enabled and not self.child_used
        estimate = self.estimate_delegation(messages, snapshot) if available else None
        decision = decide_delegation(
            snapshot,
            estimate,
            child_available=available,
            output_limit=self.child_output_tokens,
        )
        self.record("delegation_decision", phase=phase, decision=decision)
        return decision

    async def completion(self, role, messages, cap):
        session_id = self.prefix + "-" + role
        self.sessions.add(session_id)
        body = {
            "model": self.model_name.removeprefix("openai/"),
            "messages": messages,
            "tools": TOOLS[:1] if role == "child" else TOOLS,
            "parallel_tool_calls": True,
            "session_id": session_id,
            "temperature": self.temperature,
            "max_tokens": cap,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assembled = ChatCompletionStream()
        started = time.monotonic()
        self.record("request_start", role=role, cap=cap, session_id=session_id)
        ended = False
        try:
            async with self.client.stream(
                "POST",
                self.base_url + "/chat/completions",
                json=body,
                # Tool responses may be buffered; the caller owns the task deadline.
                timeout=httpx.Timeout(180, read=None),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        ended = True
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    assembled.add(chunk)
        finally:
            data = assembled.result()
            usage = data["usage"]
            if usage:
                self.context.n_input_tokens += usage["prompt_tokens"]
                self.context.n_output_tokens += usage["completion_tokens"]
                self.context.n_cache_tokens += usage.get(
                    "prompt_tokens_details", {}
                ).get("cached_tokens", 0)
                self.output_tokens[role] += usage["completion_tokens"]
            self.record(
                "completion",
                role=role,
                seconds=time.monotonic() - started,
                complete=ended,
                response=data,
            )
        if not ended or not usage:
            raise RuntimeError("Stream ended without completion and token accounting")
        choice = data["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls", "length"}:
            raise RuntimeError(f"Unexpected finish reason: {choice['finish_reason']}")
        if not choice["message"].get("tool_calls") and not choice["message"].get(
            "content"
        ):
            raise RuntimeError("Completion contains no answer or tool call")
        messages.append(choice["message"])
        return choice

    async def close_session(self, role):
        session_id = self.prefix + "-" + role
        if session_id not in self.sessions:
            return
        for attempt in range(4):
            try:
                response = await self.client.delete(
                    self.base_url + "/sessions/" + session_id
                )
                if response.status_code not in {429, 503}:
                    response.raise_for_status()
                    self.sessions.remove(session_id)
                    self.record("session_closed", role=role)
                    return
            except httpx.TransportError:
                if attempt == 3:
                    raise
            await asyncio.sleep(0.25 * 2**attempt)
        raise RuntimeError(f"Could not close {role} session")

    async def shell(self, environment, arguments, role):
        started = time.monotonic()
        command = arguments["command"]
        result = await environment.exec(command, timeout_sec=60)
        output = (
            f"exit={result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}"
        )
        if len(output) > 16000:
            output = output[:16000] + "\n[Output truncated; narrow the command.]"
        self.record(
            "tool",
            role=role,
            command=command,
            output=output,
            seconds=time.monotonic() - started,
        )
        return output

    @staticmethod
    def child_prompt(question, budget):
        if len(question.encode("utf-8")) > 4096:
            raise ValueError("Child assignment exceeds 4096 bytes; narrow the question")
        return [
            {
                "role": "system",
                "content": (
                    "You are a child investigator. Answer only the assigned question. "
                    "Use shell for read-only inspection/computation. Do not edit files, delegate, "
                    "or inspect benchmark verifier files or /tests. "
                    "Return concise supported findings and uncertainties directly when ready."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question}\nTotal output budget across all responses: {budget} tokens; "
                    "at most six model turns."
                ),
            },
        ]

    async def child_tool(self, call, environment):
        try:
            if call["function"]["name"] != "shell":
                raise ValueError(
                    "Children may only use shell for inspection/computation"
                )
            output = await self.shell(
                environment,
                json.loads(call["function"]["arguments"]),
                "child",
            )
        except (ValueError, KeyError, TypeError) as error:
            output = f"Tool error: {error}"
        reply = {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": output,
        }
        if len(json.dumps(self.child_messages + [reply]).encode()) > 16384:
            reply["content"] = (
                "Output omitted: child context budget exceeded; narrow the command."
            )
        self.child_messages.append(reply)

    async def investigate(self, environment):
        try:
            for _ in range(6):
                remaining = self.child_budget - self.output_tokens["child"]
                if remaining <= 0:
                    break
                if len(json.dumps(self.child_messages).encode()) > 16384:
                    return {"status": "context_budget_exhausted"}
                choice = await self.completion(
                    "child", self.child_messages, min(self.max_tokens, remaining)
                )
                if choice["finish_reason"] == "length":
                    break
                message = choice["message"]
                calls = message.get("tool_calls") or []
                if not calls:
                    return {
                        "status": "complete",
                        "findings": message.get("content", ""),
                    }
                for call in calls:
                    await self.child_tool(call, environment)
            return {
                "status": "budget_exhausted",
                "findings": "No complete findings; inspect child trace if needed.",
            }
        except asyncio.CancelledError:
            return {"status": "cancelled"}
        except Exception as error:
            self.record("child_error", error=str(error))
            return {"status": "error", "error": str(error)}
        finally:
            await self.close_session("child")

    def notify(self, messages):
        if self.child is None or not self.child.done() or self.child_delivered:
            return False
        result = (
            {"status": "cancelled"} if self.child.cancelled() else self.child.result()
        )
        messages.append(
            {"role": "user", "content": "Child result: " + json.dumps(result)}
        )
        self.child_delivered = True
        self.record("child_result", result=result)
        return True

    async def parent_tool(self, call, messages, environment):
        arguments = json.loads(call["function"]["arguments"])
        name = call["function"]["name"]
        if name == "shell":
            return await self.shell(environment, arguments, "parent")
        if name == "spawn":
            if not self.enabled or self.child_used:
                raise ValueError("No child budget available; work inline")
            if not all(
                isinstance(arguments[k], str) and arguments[k].strip()
                for k in ("question", "parent_command")
            ):
                raise ValueError(
                    "Provide the question and an independent parent shell command"
                )
            snapshot = await self.report("admission")
            decision = self.admission(messages, snapshot, "launch")
            if not decision["allowed"]:
                raise ValueError(
                    f"Delegation declined: {decision['reason']}; work inline"
                )
            self.child_budget = decision["estimate"]["output_tokens"]
            self.child_messages = self.child_prompt(
                arguments["question"], self.child_budget
            )
            self.child_used = True
            self.child = asyncio.create_task(self.investigate(environment))
            self.record("spawn", **arguments)
            output = await self.shell(
                environment, {"command": arguments["parent_command"]}, "parent"
            )
            return "Child started. Parent command output:\n" + output
        if name in {"wait", "cancel"}:
            if self.child is None:
                raise ValueError("No child has been started")
            started = time.monotonic()
            if name == "cancel" and not self.child.done():
                self.child.cancel()
            try:
                result = await asyncio.shield(self.child)
            except asyncio.CancelledError:
                if not self.child.cancelled():
                    raise
                result = {"status": "cancelled"}
            self.child_delivered = True
            self.record(name, seconds=time.monotonic() - started, result=result)
            return json.dumps(result)
        raise ValueError("Unknown tool")

    async def run(self, instruction, environment, context):
        self.context = context
        context.n_input_tokens = context.n_output_tokens = context.n_cache_tokens = 0
        self.started = time.monotonic()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "system",
                "content": SYSTEM
                + (
                    f"\nDelegation limit: {int(self.enabled)} child, "
                    f"{self.child_output_tokens} total child output tokens."
                ),
            },
            {"role": "user", "content": instruction},
        ]
        status = "error"
        async with httpx.AsyncClient(timeout=180, trust_env=False) as self.client:
            try:
                for turn in range(self.step_limit):
                    self.notify(messages)
                    snapshot = await self.report(turn)
                    self.admission(messages, snapshot, "report")
                    choice = await self.completion("parent", messages, self.max_tokens)
                    if choice["finish_reason"] == "length":
                        raise RuntimeError("Parent response exhausted output cap")
                    calls = choice["message"].get("tool_calls") or []
                    if not calls:
                        if self.notify(messages):
                            continue
                        if self.child is not None and not self.child.done():
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Wait for needed child findings or cancel obsolete work before finishing.",
                                }
                            )
                            continue
                        status = "final_answer"
                        break
                    for call in calls:
                        try:
                            output = await self.parent_tool(call, messages, environment)
                        except (ValueError, KeyError, TypeError) as error:
                            output = f"Tool error: {error}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": output,
                            }
                        )
                else:
                    status = "step_limit"
            finally:
                primary_error = sys.exc_info()[1]

                async def cleanup():
                    if self.child is not None:
                        if not self.child.done():
                            self.child.cancel()
                        result = (
                            await asyncio.gather(self.child, return_exceptions=True)
                        )[0]
                        if isinstance(result, Exception):
                            self.record("child_cleanup_error", error=str(result))
                    await self.close_session("child")
                    await self.close_session("parent")
                    await self.report("final")

                cleanup_task = asyncio.create_task(cleanup())
                try:
                    try:
                        await asyncio.shield(cleanup_task)
                    finally:
                        await cleanup_task
                except Exception as error:
                    self.record("cleanup_error", error=str(error))
                    if primary_error is None:
                        raise
                finally:
                    engine_output = sum(
                        s["output_tokens"] for s in self.engine_samples.values()
                    )
                    accounting_complete = engine_output >= sum(
                        self.output_tokens.values()
                    )
                    context.n_input_tokens = sum(
                        s["prompt_tokens"] for s in self.engine_samples.values()
                    )
                    context.n_output_tokens = sum(
                        s["output_tokens"] for s in self.engine_samples.values()
                    )
                    context.n_cache_tokens = sum(
                        s["reused_tokens"] for s in self.engine_samples.values()
                    )
                    context.metadata = {
                        "status": status,
                        "accounting_complete": accounting_complete,
                        "enabled": self.enabled,
                        "delegations": int(self.child_used),
                        "output_tokens": {
                            role: sum(
                                s["output_tokens"]
                                for s in self.engine_samples.values()
                                if s["session_id"] == self.prefix + "-" + role
                            )
                            for role in ("parent", "child")
                        },
                        "agent_wall_s": time.monotonic() - self.started,
                        "parent_wait_s": sum(
                            e["seconds"] for e in self.records if e["type"] == "wait"
                        ),
                    }
                    (self.logs_dir / "summary.json").write_text(
                        context.model_dump_json(indent=2)
                    )
                    (self.logs_dir / "parent.json").write_text(
                        json.dumps(messages, indent=2)
                    )
                    (self.logs_dir / "child.json").write_text(
                        json.dumps(self.child_messages, indent=2)
                    )
                    if not accounting_complete and sys.exc_info()[0] is None:
                        raise RuntimeError(
                            "Missing engine token accounting; see events.jsonl"
                        )
