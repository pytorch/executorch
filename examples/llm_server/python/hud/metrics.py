# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session totals and bounded request history, independent of terminal rendering."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Totals:
    requests: int = 0
    completed: int = 0
    errors: int = 0
    cancelled: int = 0
    reported: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefilled_tokens: int = 0
    reused_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    timed_prefill_tokens: int = 0
    timed_decode_tokens: int = 0
    ttft_seconds: float = 0.0
    ttft_count: int = 0

    def record(self, stats) -> None:
        self.reported += 1
        self.prompt_tokens += stats.prompt_tokens
        self.output_tokens += stats.completion_tokens
        self.prefilled_tokens += stats.prefilled_prompt_tokens
        self.reused_tokens += stats.reused_prompt_tokens
        if stats.prefill_ms > 0:
            self.prefill_ms += stats.prefill_ms
            self.timed_prefill_tokens += stats.prefilled_prompt_tokens
        if stats.decode_ms > 0:
            self.decode_ms += stats.decode_ms
            self.timed_decode_tokens += stats.completion_tokens

    def finish(self, turn: Turn) -> None:
        if turn.state == "error":
            self.errors += 1
        elif turn.state == "cancelled":
            self.cancelled += 1
        else:
            self.completed += 1
        if turn.first_output is not None:
            self.ttft_seconds += turn.first_output - turn.started
            self.ttft_count += 1


@dataclass
class Session:
    totals: Totals = field(default_factory=Totals)
    context: Optional[int] = None
    context_request: int = 0
    context_epoch: int = 0
    last_request: int = 0
    closed: bool = False


@dataclass
class Turn:
    number: int
    session_id: Optional[str]
    started: float
    session: Session = field(repr=False)
    server_totals: Totals = field(repr=False)
    metrics: Metrics = field(repr=False)
    state: str = "waiting"
    ended: Optional[float] = None
    first_output: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    prefill_rate: Optional[float] = None
    decode_rate: Optional[float] = None
    reused_tokens: Optional[int] = None
    error: Optional[str] = None
    context_epoch: int = 0

    def activate(self) -> None:
        if self.session.closed:
            previous = self.session
            session = self.metrics.sessions.get(self.session_id)
            if session is None or session.closed:
                session = Session()
                self.metrics.sessions[self.session_id] = session
            previous.totals.requests -= 1
            session.totals.requests += 1
            session.last_request = max(session.last_request, self.number)
            self.session = session
        self.context_epoch = self.session.context_epoch

    def output(self, arrived: float) -> None:
        if self.first_output is None:
            self.first_output = arrived
        if self.ended is None:
            self.state = "generating"

    def record_stats(self, stats) -> None:
        if not stats.terminal_received or self.prompt_tokens is not None:
            return
        self.prompt_tokens = stats.prompt_tokens
        self.output_tokens = stats.completion_tokens
        self.reused_tokens = stats.reused_prompt_tokens
        self.prefill_rate = stats.prefill_tok_s if stats.prefill_ms > 0 else None
        self.decode_rate = stats.decode_tok_s if stats.decode_ms > 0 else None
        # A cancelled HTTP request can settle in the worker after its row is gone.
        self.server_totals.record(stats)
        self.session.totals.record(stats)
        if (
            self.context_epoch == self.session.context_epoch
            and self.number > self.session.context_request
        ):
            self.session.context = self.prompt_tokens + self.output_tokens
            self.session.context_request = self.number


def record_error(turn: Optional[Turn], category: str) -> None:
    if turn is not None and turn.error is None:
        turn.error = category


class Metrics:
    def __init__(self, history_size: int = 32):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.history_size = history_size
        self.started = time.monotonic()
        self.totals = Totals()
        self.sessions: OrderedDict[Optional[str], Session] = OrderedDict()
        self.turns: OrderedDict[int, Turn] = OrderedDict()
        self._sequence = 0

    @property
    def completed(self):
        return self.totals.completed

    @property
    def errors(self):
        return self.totals.errors

    @property
    def cancelled(self):
        return self.totals.cancelled

    def begin(self, session_id: Optional[str]) -> Turn:
        session = self.sessions.get(session_id)
        if session is None or session.closed:
            session = Session()
            self.sessions[session_id] = session
        self.sessions.move_to_end(session_id)
        self._sequence += 1
        session.last_request = self._sequence
        session.totals.requests += 1
        self.totals.requests += 1
        turn = Turn(
            self._sequence, session_id, time.monotonic(), session, self.totals, self
        )
        self.turns[turn.number] = turn
        return turn

    def finish(self, turn: Turn, state: str = "done") -> None:
        if turn.ended is not None:
            return
        turn.ended = time.monotonic()
        if turn.error is not None:
            turn.state = "error"
        elif state == "cancelled":
            turn.state = "cancelled"
        else:
            turn.state = "done"
        self.totals.finish(turn)
        turn.session.totals.finish(turn)
        if turn.session.closed:
            self.turns.pop(turn.number, None)
            if self.sessions.get(turn.session_id) is turn.session and not any(
                row.session is turn.session for row in self.turns.values()
            ):
                self.sessions.pop(turn.session_id)
        finished = sorted(
            (
                key
                for key, row in self.turns.items()
                if row.session is turn.session and row.ended is not None
            ),
            key=lambda key: self.turns[key].ended,
        )
        for key in finished[: -min(3, self.history_size)]:
            del self.turns[key]
        active = {row.session_id for row in self.turns.values() if row.ended is None}
        idle = [key for key in self.sessions if key not in active]
        for session_id in idle[: -self.history_size]:
            self.close_session(session_id)

    def close_session(self, session_id: Optional[str]) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session.closed = True
        for key, turn in list(self.turns.items()):
            if turn.session is session and turn.ended is not None:
                del self.turns[key]
        if not any(row.session is session for row in self.turns.values()):
            self.sessions.pop(session_id)

    def reset_context(self, session_id: str, value: Optional[int] = 0) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session.context = value
            session.context_request = 0
            session.context_epoch += 1
