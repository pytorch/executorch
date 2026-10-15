# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental one-child agent; inline mode uses the same tools and loop."""

import asyncio
import json
import math
import sys
import time
import uuid
from pathlib import Path

import httpx
from harbor.agents.base import BaseAgent
from tokenizers import Tokenizer

from .delegation import decide_delegation
from .inference_cost import InferenceCost
from .streaming import ChatCompletionStream

SYSTEM = """Solve the user's task in the container. Inspect before editing, use tools for evidence,
verify the result, then give a concise final answer. Do not inspect benchmark verifier files or /tests.
The parent owns edits and final verification. A delegated child investigates only its assigned
question without modifying files or delegating. Do not edit files a running child is inspecting.

Optimize time to verified correct completion, including all parent and child work.
Generation batching shares parameter reads; each sequence adds computation and KV-cache reads.
A larger generation batch can improve aggregate throughput while increasing per-token latency.
The in-flight request limit is not the generation batch size.
Longer context increases KV-cache reads; sliding-window attention can cap this cost.
Prefill can delay ongoing generation. Prefix caching avoids repeated prefill, not later attention cost.
Shorter context does not necessarily free allocated memory.

The harness checks configured limits and logs inference costs for measurement.
If a spawn is declined, continue inline without retrying delegation.
Do not infer permission from idle hardware or invent hardware rates or execution times.
Available decode slots do not imply free capacity.

Before spawning: choose one necessary independent question and a parent shell action you still need to run.
Pass that action as parent_command; the harness runs it alongside the child and returns its output.
Provide expected_output_tokens for the child's tool arguments, intermediate responses and final
findings across all turns, excluding thinking. The harness estimates thinking separately and
enforces a total output cap covering both.
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
                "description": "A necessary shell action you still need to run, independent of the child's question.",
            },
            "expected_output_tokens": {
                "type": "integer",
                "minimum": 1,
                "description": "Expected child tool arguments and answer text across all turns, excluding thinking; not the total output cap.",
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
        hf_tokenizer=None,
        enabled=False,
        max_tokens=4096,
        child_output_tokens=1536,
        step_limit=100,
        temperature=0.8,
        calibration_path=None,
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
        self.child_expected_output_tokens = None
        self.child_expected_thinking_tokens = None
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
        self.cost_model = None
        self.child_prompt_tokens = None
        self.last_step_id = None
        if calibration_path:
            calibration = json.loads(Path(calibration_path).read_text())
            if calibration["model_id"] != self.model_name.removeprefix("openai/"):
                raise ValueError("Calibration model does not match the agent model")
            self.cost_model = InferenceCost.from_dict(calibration["cost"])
        self.started = time.monotonic()
        self.prefix = str(uuid.uuid4())
        self.client = None
        self.context = None
        self.observed_thinking_tokens = 0
        self.observed_non_thinking_tokens = 0
        self.tokenizer = None
        if hf_tokenizer:
            path = Path(hf_tokenizer).expanduser()
            if path.is_dir():
                path /= "tokenizer.json"
            self.tokenizer = (
                Tokenizer.from_file(str(path))
                if path.is_file()
                else Tokenizer.from_pretrained(hf_tokenizer)
            )

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
        steps = (snapshot.get("engine") or {}).get("recent_steps", [])
        for step in steps:
            if (
                self.cost_model is not None
                and self.last_step_id is not None
                and step["id"] > self.last_step_id
                and step["succeeded"]
            ):
                try:
                    predicted_ms = self.cost_model.step(step["inputs"]) * 1000
                except ValueError as error:
                    self.record("inference_estimate_unavailable", error=str(error))
                else:
                    self.record(
                        "inference_step",
                        step_id=step["id"],
                        inputs=step["inputs"],
                        predicted_ms=predicted_ms,
                        measured_ms=step["elapsed_ms"],
                    )
        self.last_step_id = max(
            (step["id"] for step in steps), default=self.last_step_id or 0
        )
        samples = snapshot["recent_generations"]
        for sample in samples:
            if sample["session_id"] in {
                self.prefix + "-parent",
                self.prefix + "-child",
            }:
                self.engine_samples[sample["id"]] = sample
        return snapshot

    def estimate_child_inference(self, snapshot):
        """Estimate cold prefill and decode steps at the current prompt lengths."""
        cost = self.cost_model
        if cost is None or self.child_prompt_tokens is None:
            return None
        child = {"position": self.child_prompt_tokens, "tokens": 1, "decode": True}
        positions = {
            s["session_id"]: s["committed_position"]
            for s in (snapshot.get("engine") or {}).get("sessions", [])
        }
        parent_position = positions.get(self.prefix + "-parent")
        paired = None
        if type(parent_position) is int and parent_position >= 0:
            paired = cost.step(
                [child, {"position": parent_position, "tokens": 1, "decode": True}]
            )
        return {
            "child_prompt_tokens": self.child_prompt_tokens,
            "parent_position": parent_position,
            "cold_prefill_s": cost.prefill(0, self.child_prompt_tokens),
            "solo_decode_step_s": cost.step([child]),
            "paired_decode_step_s": paired,
        }

    def estimate_thinking_tokens(self, messages, snapshot, proposal) -> int | None:
        """Scale expected output by the observed ratio of thinking to other tokens."""
        if not self.observed_non_thinking_tokens:
            return None
        return math.ceil(
            proposal["expected_output_tokens"]
            * self.observed_thinking_tokens
            / self.observed_non_thinking_tokens
        )

    def observe_thinking_tokens(self, response, role):
        choice = response["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            return
        usage = response["usage"]
        thinking = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        source = "usage"
        if thinking is None and self.tokenizer is not None:
            thinking = len(
                self.tokenizer.encode(
                    choice["message"].get("reasoning_content") or "",
                    add_special_tokens=False,
                ).ids
            )
            source = "tokenizer"
        if type(thinking) is not int or not 0 <= thinking <= usage["completion_tokens"]:
            return
        other = usage["completion_tokens"] - thinking
        self.observed_thinking_tokens += thinking
        self.observed_non_thinking_tokens += other
        self.record(
            "thinking_observation",
            role=role,
            thinking_tokens=thinking,
            non_thinking_tokens=other,
            source=source,
        )

    def admission(self, messages, snapshot, proposal):
        decision = decide_delegation(
            snapshot,
            child_available=self.enabled and not self.child_used,
            expected_output_tokens=proposal.get("expected_output_tokens"),
            output_limit=self.child_output_tokens,
        )
        thinking_tokens = (
            self.estimate_thinking_tokens(messages, snapshot, proposal)
            if decision["allowed"]
            else None
        )
        decision["expected_thinking_tokens"] = thinking_tokens
        decision["expected_total_tokens"] = (
            proposal["expected_output_tokens"] + thinking_tokens
            if thinking_tokens is not None
            else None
        )
        self.record(
            "delegation_decision",
            phase="launch",
            policy="limits_only",
            proposal=proposal,
            observed_thinking_tokens=self.observed_thinking_tokens,
            observed_non_thinking_tokens=self.observed_non_thinking_tokens,
            inference_estimate=self.estimate_child_inference(snapshot),
            decision=decision,
        )
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
            if choice["finish_reason"] == "length":
                raise RuntimeError(
                    f"Reasoning exhausted the {cap}-token response cap without "
                    "an answer or tool call"
                )
            raise RuntimeError("Completion contains no answer or tool call")
        self.observe_thinking_tokens(data, role)
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
        result = {
            "status": "budget_exhausted",
            "findings": "No complete findings; inspect child trace if needed.",
        }
        try:
            for _ in range(6):
                remaining = self.child_output_tokens - self.output_tokens["child"]
                if remaining <= 0:
                    break
                if len(json.dumps(self.child_messages).encode()) > 16384:
                    result = {"status": "context_budget_exhausted"}
                    break
                choice = await self.completion(
                    "child", self.child_messages, min(self.max_tokens, remaining)
                )
                if choice["finish_reason"] == "length":
                    break
                message = choice["message"]
                calls = message.get("tool_calls") or []
                if not calls:
                    result = {
                        "status": "complete",
                        "findings": message.get("content", ""),
                    }
                    break
                for call in calls:
                    await self.child_tool(call, environment)
        except asyncio.CancelledError:
            result = {"status": "cancelled"}
        except Exception as error:
            self.record("child_error", error=str(error))
            result = {"status": "error", "error": str(error)}
        finally:
            self.record(
                "child_output",
                expected_output_tokens=self.child_expected_output_tokens,
                expected_thinking_tokens=self.child_expected_thinking_tokens,
                expected_total_tokens=(
                    self.child_expected_output_tokens
                    + self.child_expected_thinking_tokens
                    if self.child_expected_output_tokens is not None
                    and self.child_expected_thinking_tokens is not None
                    else None
                ),
                actual_output_tokens=self.output_tokens["child"],
                output_limit=self.child_output_tokens,
                status=result["status"],
            )
            await self.close_session("child")
        return result

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

    async def count_child_prompt(self, question):
        self.child_prompt_tokens = None
        if self.cost_model is None:
            return
        try:
            response = await self.client.post(
                self.base_url + "/chat/tokenize",
                json={
                    "model": self.model_name.removeprefix("openai/"),
                    "messages": self.child_prompt(question, self.child_output_tokens),
                    "tools": TOOLS[:1],
                },
            )
            response.raise_for_status()
            count = response.json()["prompt_tokens"]
            self.child_prompt_tokens = (
                count if type(count) is int and count > 0 else None
            )
        except (httpx.HTTPError, KeyError, ValueError) as error:
            self.record("inference_estimate_unavailable", error=str(error))

    async def parent_tool(self, call, messages, environment):
        arguments = json.loads(call["function"]["arguments"])
        name = call["function"]["name"]
        if name == "shell":
            return await self.shell(environment, arguments, "parent")
        if name == "spawn":
            if not self.enabled or self.child_used:
                raise ValueError("No child budget available; work inline")
            if not all(
                isinstance(arguments.get(k), str) and arguments[k].strip()
                for k in ("question", "parent_command")
            ):
                raise ValueError("Provide a question and an independent parent command")
            self.child_prompt(arguments["question"], self.child_output_tokens)
            await self.count_child_prompt(arguments["question"])
            snapshot = await self.report("admission")
            decision = self.admission(messages, snapshot, arguments)
            if not decision["allowed"]:
                raise ValueError(
                    f"Delegation declined: {decision['reason']}; work inline"
                )
            self.child_messages = self.child_prompt(
                arguments["question"], self.child_output_tokens
            )
            self.child_expected_output_tokens = arguments["expected_output_tokens"]
            self.child_expected_thinking_tokens = decision["expected_thinking_tokens"]
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
        self.record(
            "inference_calibration",
            cost=self.cost_model.to_dict() if self.cost_model is not None else None,
        )
        async with httpx.AsyncClient(timeout=180, trust_env=False) as self.client:
            try:
                for turn in range(self.step_limit):
                    self.notify(messages)
                    await self.report(turn)
                    choice = await self.completion("parent", messages, self.max_tokens)
                    if choice["finish_reason"] == "length":
                        raise RuntimeError("Parent response exhausted output cap")
                    calls = choice["message"].get("tool_calls") or []
                    self.record(
                        "action_selection",
                        turn=turn,
                        tools=[call["function"]["name"] for call in calls],
                        delegation_proposed=any(
                            call["function"]["name"] == "spawn" for call in calls
                        ),
                        child_available=self.enabled and not self.child_used,
                        child_active=self.child is not None and not self.child.done(),
                    )
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
                            self.record(
                                "tool_error",
                                turn=turn,
                                tool=call["function"]["name"],
                                error=str(error),
                            )
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
