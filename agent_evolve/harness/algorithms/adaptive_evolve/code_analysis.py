"""Code-execution-aware trajectory analyzer.

Extracts rich signals about HOW code execution was used in each task,
not just whether it was used. This gives the evolver LLM enough
information to create skills that teach the hybrid pattern:
direct calls for reasoning → code execution for data processing.

Originally from code_evolve; inlined here so adaptive_evolve is self-contained.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .base_analysis import BatchAnalysis


# ── Execution pattern categories ─────────────────────────────

PATTERN_DIRECT_ONLY = "direct_only"
PATTERN_CODE_ONLY = "code_only"
PATTERN_HYBRID_GOOD = "hybrid_reason_then_code"
PATTERN_HYBRID_BAD = "hybrid_code_first"
PATTERN_HYBRID_MIXED = "hybrid_mixed"

HIGH_TOOL_CALL_THRESHOLD = 15


@dataclass
class TaskCodeProfile:
    """Per-task analysis of how code execution was used."""
    task_id: str
    score: float
    success: bool
    pattern: str
    total_tool_calls: int
    direct_calls_before_code: int
    code_exec_calls: int
    code_exec_errors: int
    code_exec_position: str  # "early" | "mid" | "late" | "none"
    tools_used: list[str] = field(default_factory=list)


@dataclass
class CodeExecStats:
    """Batch-level statistics about code execution usage."""
    total_tasks: int = 0
    tasks_used_code: int = 0
    tasks_no_code: int = 0

    code_pass: int = 0
    code_fail: int = 0
    no_code_pass: int = 0
    no_code_fail: int = 0

    pattern_counts: dict[str, int] = field(default_factory=dict)
    pattern_pass_rates: dict[str, float] = field(default_factory=dict)

    failed_profiles: list[TaskCodeProfile] = field(default_factory=list)
    missed_opportunities: list[dict[str, Any]] = field(default_factory=list)
    effective_patterns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def code_pass_rate(self) -> float:
        total = self.code_pass + self.code_fail
        return self.code_pass / total if total else 0.0

    @property
    def no_code_pass_rate(self) -> float:
        total = self.no_code_pass + self.no_code_fail
        return self.no_code_pass / total if total else 0.0

    def summary_text(self) -> str:
        parts = [f"code_exec: {self.tasks_used_code}/{self.total_tasks} tasks"]
        if self.tasks_used_code:
            parts.append(f"code_pass_rate={self.code_pass_rate:.0%}")
        if self.tasks_no_code:
            parts.append(f"no_code_pass_rate={self.no_code_pass_rate:.0%}")
        if self.pattern_counts:
            top = sorted(self.pattern_counts.items(), key=lambda x: -x[1])[:3]
            parts.append("patterns: " + ", ".join(f"{p}={c}" for p, c in top))
        if self.missed_opportunities:
            parts.append(f"missed={len(self.missed_opportunities)}")
        return " | ".join(parts)


class CodeExecAnalyzer:
    """Analyzes trajectories for code execution patterns with rich signals."""

    def analyze(
        self, logs: list[dict[str, Any]], analysis: BatchAnalysis
    ) -> CodeExecStats:
        stats = CodeExecStats(total_tasks=len(logs))
        pattern_pass: Counter[str] = Counter()
        pattern_total: Counter[str] = Counter()

        for log in logs:
            profile = self._build_profile(log)

            if profile.code_exec_calls > 0:
                stats.tasks_used_code += 1
                if profile.success:
                    stats.code_pass += 1
                else:
                    stats.code_fail += 1
            else:
                stats.tasks_no_code += 1
                if profile.success:
                    stats.no_code_pass += 1
                else:
                    stats.no_code_fail += 1

            pattern_total[profile.pattern] += 1
            if profile.success:
                pattern_pass[profile.pattern] += 1

            if not profile.success:
                stats.failed_profiles.append(profile)

                if (profile.code_exec_calls == 0
                        and profile.total_tool_calls >= HIGH_TOOL_CALL_THRESHOLD):
                    repeated = self._find_repeated_tools(log.get("steps", []))
                    stats.missed_opportunities.append({
                        "task_id": profile.task_id,
                        "score": profile.score,
                        "tool_calls": profile.total_tool_calls,
                        "repeated_tools": repeated,
                        "tools": profile.tools_used[:5],
                    })

            if profile.success and profile.code_exec_calls > 0:
                stats.effective_patterns.append({
                    "task_id": profile.task_id,
                    "score": profile.score,
                    "pattern": profile.pattern,
                    "direct_before_code": profile.direct_calls_before_code,
                    "tools": profile.tools_used[:5],
                })

        stats.pattern_counts = dict(pattern_total)
        for pattern in pattern_total:
            total = pattern_total[pattern]
            passed = pattern_pass.get(pattern, 0)
            stats.pattern_pass_rates[pattern] = passed / total if total else 0.0

        return stats

    def _build_profile(self, log: dict[str, Any]) -> TaskCodeProfile:
        task_id = log.get("task_id", "")
        success = log.get("success", False)
        score = log.get("score", 0.0)
        steps = log.get("steps", [])

        total_calls = 0
        code_exec_calls = 0
        code_exec_errors = 0
        direct_calls_before_code = 0
        first_code_exec_index = -1
        tools_used: list[str] = []
        seen_tools: set[str] = set()

        for step in steps:
            for tc in step.get("tool_calls", []):
                tool = tc.get("tool", "")
                total_calls += 1

                if tool not in seen_tools:
                    tools_used.append(tool)
                    seen_tools.add(tool)

                if tool in ("execute_code", "mcp-code-executor_execute_code"):
                    code_exec_calls += 1
                    if first_code_exec_index < 0:
                        first_code_exec_index = total_calls
                else:
                    if first_code_exec_index < 0:
                        direct_calls_before_code += 1

            for tr in step.get("tool_results", []):
                for item in (tr.get("content", []) if isinstance(tr.get("content"), list) else []):
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        if "Error:" in text and any(
                            tc.get("tool") in ("execute_code", "mcp-code-executor_execute_code")
                            for tc in step.get("tool_calls", [])
                        ):
                            code_exec_errors += 1

        pattern = self._classify_pattern(
            code_exec_calls, total_calls, direct_calls_before_code, first_code_exec_index
        )

        if code_exec_calls == 0:
            position = "none"
        elif first_code_exec_index <= 2:
            position = "early"
        elif first_code_exec_index <= total_calls * 0.5:
            position = "mid"
        else:
            position = "late"

        return TaskCodeProfile(
            task_id=task_id,
            score=score,
            success=success,
            pattern=pattern,
            total_tool_calls=total_calls,
            direct_calls_before_code=direct_calls_before_code,
            code_exec_calls=code_exec_calls,
            code_exec_errors=code_exec_errors,
            code_exec_position=position,
            tools_used=tools_used,
        )

    def _classify_pattern(
        self, code_calls: int, total_calls: int,
        direct_before: int, first_code_idx: int
    ) -> str:
        if code_calls == 0:
            return PATTERN_DIRECT_ONLY
        if total_calls == code_calls:
            return PATTERN_CODE_ONLY

        direct_after = total_calls - code_calls - direct_before
        if direct_before >= 2 and first_code_idx > 2:
            if direct_after <= 2:
                return PATTERN_HYBRID_GOOD
            return PATTERN_HYBRID_MIXED
        if first_code_idx <= 2:
            return PATTERN_HYBRID_BAD
        return PATTERN_HYBRID_MIXED

    def _find_repeated_tools(self, steps: list[dict]) -> list[dict]:
        counts: Counter[str] = Counter()
        for step in steps:
            for tc in step.get("tool_calls", []):
                counts[tc.get("tool", "")] += 1
        return [
            {"tool": t, "count": c}
            for t, c in counts.most_common(5)
            if c >= 3
        ]
