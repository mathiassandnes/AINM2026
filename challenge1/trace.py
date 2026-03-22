"""
Task tracing — collects all events for a task and emits a complete trace.

Usage:
    trace = TaskTrace(task_id, prompt, task_type)
    trace.phase1(steps)
    trace.phase2(params)
    trace.exec_step(i, method, path, params, body, response, ok)
    trace.fallback_call(turn, tool, args, result, ok)
    trace.emit()  # logs the full trace
"""

import json
import logging
import time

log = logging.getLogger("trace")


class TaskTrace:
    def __init__(self, task_id: str, prompt: str, task_type: str):
        self.task_id = task_id
        self.prompt = prompt
        self.task_type = task_type
        self.t0 = time.time()
        self.events: list[str] = []
        self.api_calls = 0
        self.api_errors = 0
        self.outcome = "unknown"

    def _add(self, line: str):
        self.events.append(line)

    def phase1(self, steps: list[dict] | None, error: str = ""):
        if error:
            self._add(f"PHASE1: FAILED — {error}")
            return
        if steps is None:
            self._add("PHASE1: no steps returned")
            return
        self._add(f"PHASE1: {len(steps)} operations selected")
        for i, s in enumerate(steps):
            fm = ""
            if s.get("find_match"):
                fm = f" [find_match: {s['find_match'].get('field','?')} contains '{s['find_match'].get('contains','?')}']"
            pp = ""
            if s.get("path_params"):
                pp = f" [path_params: {json.dumps(s['path_params'], default=str)}]"
            self._add(f"  step_{i}: {s['operation']}{fm}{pp} — {s.get('description', '')[:120]}")

    def phase2(self, params: dict | None, error: str = ""):
        if error:
            self._add(f"PHASE2: FAILED — {error}")
            return
        if params is None:
            self._add("PHASE2: no params returned")
            return
        self._add(f"PHASE2: filled params for {len(params)} steps")
        for key, val in params.items():
            val_str = json.dumps(val, ensure_ascii=False, default=str)
            if len(val_str) > 300:
                val_str = val_str[:300] + "..."
            self._add(f"  {key}: {val_str}")

    def exec_step(self, i: int, total: int, op_id: str, method: str, path: str,
                  params: dict | None, body: dict | None,
                  response: dict | None, ok: bool, find_match_result: str = ""):
        self.api_calls += 1
        if not ok:
            self.api_errors += 1

        status = "OK" if ok else "FAIL"
        line = f"  EXEC step_{i}/{total}: {status} {method} {path}"

        # Show params if non-default
        if params:
            p_str = json.dumps(params, ensure_ascii=False, default=str)
            if len(p_str) > 200:
                p_str = p_str[:200] + "..."
            line += f"\n    params: {p_str}"

        if body:
            b_str = json.dumps(body, ensure_ascii=False, default=str)
            if len(b_str) > 400:
                b_str = b_str[:400] + "..."
            line += f"\n    body: {b_str}"

        if response:
            r_str = json.dumps(response, ensure_ascii=False, default=str)
            if len(r_str) > 400:
                r_str = r_str[:400] + "..."
            line += f"\n    response: {r_str}"

        if find_match_result:
            line += f"\n    find_match: {find_match_result}"

        self._add(line)

    def exec_error(self, i: int, total: int, op_id: str, error: str):
        self._add(f"  EXEC step_{i}/{total}: EXCEPTION {op_id} — {error[:200]}")

    def exec_result(self, success: bool, completed: int, total: int, error: str = ""):
        if success:
            self._add(f"EXEC: SUCCESS {completed}/{total} steps")
            self.outcome = "plan_success"
        else:
            self._add(f"EXEC: FAILED at step {completed}/{total} — {error[:200]}")
            self.outcome = "plan_failed"

    def fallback_start(self):
        self._add("FALLBACK: starting reactive loop")

    def fallback_llm(self, turn: int, text: str):
        text_short = text[:200].replace("\n", " ")
        self._add(f"  FB turn {turn}: LLM says: {text_short}")

    def fallback_call(self, turn: int, tool: str, args: dict, result: str, ok: bool):
        self.api_calls += 1
        if not ok:
            self.api_errors += 1

        status = "OK" if ok else "FAIL"
        args_str = json.dumps(args, ensure_ascii=False, default=str)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        result_short = result[:300] if ok else result[:400]

        self._add(f"  FB turn {turn}: {status} {tool}({args_str})\n    → {result_short}")

    def fallback_end(self, turns: int, tools_ok: int, tools_err: int):
        self._add(f"FALLBACK: done in {turns} turns, {tools_ok} ok + {tools_err} err tool calls")
        if self.outcome == "plan_failed":
            self.outcome = "fallback"
        elif self.outcome == "unknown":
            self.outcome = "fallback_only"

    def emit(self, api_calls: int = 0, api_errors: int = 0):
        """Emit the full trace as structured log entries."""
        duration = time.time() - self.t0

        # Use the actual client counts if provided
        if api_calls:
            self.api_calls = api_calls
        if api_errors:
            self.api_errors = api_errors

        # Build the full trace text
        header = (
            f"═══ TASK TRACE [{self.task_id}] ═══\n"
            f"Type: {self.task_type}\n"
            f"Outcome: {self.outcome}\n"
            f"Duration: {duration:.1f}s | API: {self.api_calls} calls, {self.api_errors} errors\n"
            f"Prompt: {self.prompt}\n"
            f"{'─' * 60}"
        )

        trace_body = "\n".join(self.events)
        full_trace = f"{header}\n{trace_body}\n{'═' * 60}"

        # Cloud Logging truncates at ~256KB. Split into chunks if needed.
        # Each chunk gets the task_id for filtering.
        max_chunk = 50000  # ~50KB per log entry, safe margin
        extra = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "outcome": self.outcome,
            "duration_s": round(duration, 1),
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
        }

        if len(full_trace) <= max_chunk:
            log.info(full_trace, extra=extra)
        else:
            # Split: header + events in chunks
            log.info(header, extra={**extra, "chunk": "0/N"})
            chunk = ""
            chunk_idx = 1
            for event in self.events:
                if len(chunk) + len(event) + 1 > max_chunk:
                    log.info(f"[{self.task_id}] trace chunk {chunk_idx}:\n{chunk}",
                             extra={**extra, "chunk": f"{chunk_idx}/N"})
                    chunk = ""
                    chunk_idx += 1
                chunk += event + "\n"
            if chunk:
                log.info(f"[{self.task_id}] trace chunk {chunk_idx} (final):\n{chunk}",
                         extra={**extra, "chunk": f"{chunk_idx}/N"})
