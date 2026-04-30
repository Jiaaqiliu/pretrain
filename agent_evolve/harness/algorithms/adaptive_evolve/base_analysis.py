"""Base trajectory analysis — batch-level error extraction and statistics.

Provides the foundational analysis layer: tool-calling errors, strategy
issues, hallucination detection, and score distribution. Higher-level
analyzers (AdaptiveAnalyzer, CodeExecAnalyzer) build on top of this.

Originally from mcp_evolve; inlined here so adaptive_evolve is self-contained.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ── Protocols ────────────────────────────────────────────────


@runtime_checkable
class ErrorPatternExtractor(Protocol):
    """Extracts tool-calling errors from a single text block + step context."""

    def extract(
        self,
        text: str,
        step: dict[str, Any],
        task_id: str,
        analysis: Any,
    ) -> None: ...


@runtime_checkable
class AutoCorrector(Protocol):
    """Applies deterministic fixes to the workspace based on analysis results."""

    def apply(
        self,
        workspace: Any,
        analysis: Any,
        accumulated_state: dict[str, Any],
    ) -> int: ...


# ── Data classes ─────────────────────────────────────────────


@dataclass
class ToolError:
    """A tool-calling error extracted from a trajectory."""
    task_id: str
    error_type: str
    tool_called: str
    detail: str = ""
    available_tools: list[str] = field(default_factory=list)
    correct_tool: str = ""


@dataclass
class StrategyIssue:
    """A higher-level strategy or reasoning issue."""
    task_id: str
    issue_type: str
    detail: str = ""


@dataclass
class BatchAnalysis:
    """Aggregated analysis of a batch of observations."""
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0

    tool_errors: list[ToolError] = field(default_factory=list)
    hallucination_map: dict[str, str] = field(default_factory=dict)
    param_errors: list[dict[str, Any]] = field(default_factory=list)

    strategy_issues: list[StrategyIssue] = field(default_factory=list)

    tool_error_counts: dict[str, int] = field(default_factory=dict)
    strategy_issue_counts: dict[str, int] = field(default_factory=dict)
    failed_tool_freq: dict[str, int] = field(default_factory=dict)

    partial_scores: list[dict[str, Any]] = field(default_factory=list)
    score_buckets: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tasks if self.total_tasks else 0.0

    def summary_text(self) -> str:
        parts = [
            f"{self.total_tasks} tasks ({self.passed} pass, {self.failed} fail, "
            f"{self.pass_rate:.0%} rate)",
        ]
        if self.tool_error_counts:
            parts.append("tool errors: " + ", ".join(
                f"{t}={c}" for t, c in self.tool_error_counts.items()
            ))
        if self.strategy_issue_counts:
            parts.append("strategy issues: " + ", ".join(
                f"{t}={c}" for t, c in self.strategy_issue_counts.items()
            ))
        return " | ".join(parts)


# ── Main entry point ─────────────────────────────────────────


def analyze_observations(
    logs: list[dict[str, Any]],
    error_extractor: ErrorPatternExtractor | None = None,
) -> BatchAnalysis:
    """Analyze observation logs to extract structured failure patterns."""
    if error_extractor is None:
        error_extractor = McpErrorPatternExtractor()

    analysis = BatchAnalysis(total_tasks=len(logs))
    scores: list[float] = []

    for log in logs:
        score = log.get("score", 0.0)
        scores.append(score)

        if log.get("success"):
            analysis.passed += 1
            continue
        analysis.failed += 1

        task_id = log.get("task_id", "")
        steps = log.get("steps", [])
        feedback = log.get("feedback_detail", "")
        output = log.get("agent_output", "")

        _extract_tool_errors(steps, task_id, analysis, error_extractor)
        _extract_strategy_issues(steps, task_id, feedback, output, analysis)

        if 0 < score < 1.0:
            analysis.partial_scores.append({
                "task_id": task_id,
                "score": score,
                "feedback": feedback[:500],
            })

    # Aggregate statistics
    for err in analysis.tool_errors:
        analysis.tool_error_counts[err.error_type] = (
            analysis.tool_error_counts.get(err.error_type, 0) + 1
        )
        analysis.failed_tool_freq[err.tool_called] = (
            analysis.failed_tool_freq.get(err.tool_called, 0) + 1
        )

    for issue in analysis.strategy_issues:
        analysis.strategy_issue_counts[issue.issue_type] = (
            analysis.strategy_issue_counts.get(issue.issue_type, 0) + 1
        )

    for err in analysis.tool_errors:
        if err.error_type == "hallucinated_name" and err.correct_tool:
            analysis.hallucination_map[err.tool_called] = err.correct_tool

    for s in scores:
        bucket = f"{int(s * 5) / 5:.1f}-{int(s * 5) / 5 + 0.2:.1f}"
        analysis.score_buckets[bucket] = analysis.score_buckets.get(bucket, 0) + 1

    return analysis


# ── Tool error extraction ────────────────────────────────────


def _extract_tool_errors(
    steps: list[dict[str, Any]],
    task_id: str,
    analysis: BatchAnalysis,
    extractor: ErrorPatternExtractor,
) -> None:
    for step in steps:
        for tr in step.get("tool_results", []):
            content = tr.get("content", [])
            for item in (content if isinstance(content, list) else [content]):
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                extractor.extract(text, step, task_id, analysis)

        error = step.get("error", "")
        if error:
            extractor.extract(error, step, task_id, analysis)


# ── Strategy issue extraction ────────────────────────────────


def _extract_strategy_issues(
    steps: list[dict[str, Any]],
    task_id: str,
    feedback: str,
    output: str,
    analysis: BatchAnalysis,
) -> None:
    if not output or not output.strip():
        analysis.strategy_issues.append(StrategyIssue(
            task_id=task_id,
            issue_type="empty_output",
            detail="Agent produced empty or near-empty output",
        ))

    for step in steps:
        error = step.get("error", "")
        text = step.get("text", "")
        combined = error + text
        if "context window overflow" in combined.lower():
            analysis.strategy_issues.append(StrategyIssue(
                task_id=task_id,
                issue_type="context_overflow",
                detail="Agent hit context window limit",
            ))
            break

    for step in steps:
        error = step.get("error", "")
        if "timeout" in error.lower() or "timed out" in error.lower():
            analysis.strategy_issues.append(StrategyIssue(
                task_id=task_id,
                issue_type="timeout",
                detail=error[:200],
            ))
            break

    tool_call_counts: Counter[str] = Counter()
    tool_error_counts: Counter[str] = Counter()
    for step in steps:
        for tc in step.get("tool_calls", []):
            name = tc.get("tool", "")
            tool_call_counts[name] += 1
        for tr in step.get("tool_results", []):
            status = tr.get("status", "")
            if status == "error":
                for tc in step.get("tool_calls", []):
                    tool_error_counts[tc.get("tool", "")] += 1

    for tool, err_count in tool_error_counts.items():
        if err_count >= 3:
            analysis.strategy_issues.append(StrategyIssue(
                task_id=task_id,
                issue_type="repeated_failure",
                detail=f"Tool '{tool}' failed {err_count} times in this task",
            ))

    wrong_source_keywords = [
        "wrong paper", "wrong database", "wrong source",
        "did not use", "never called", "should have used",
    ]
    feedback_lower = feedback.lower()
    if any(kw in feedback_lower for kw in wrong_source_keywords):
        analysis.strategy_issues.append(StrategyIssue(
            task_id=task_id,
            issue_type="wrong_tool_choice",
            detail=feedback[:300],
        ))


# ── Default MCP error extractor ──────────────────────────────


import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class McpErrorPatternExtractor:
    """Extracts tool errors from Strands/MCP-Atlas error messages."""

    def extract(
        self,
        text: str,
        step: dict[str, Any],
        task_id: str,
        analysis: Any,
    ) -> None:
        import re as _re

        # Pattern 1: tool_name=<X>, available_tools=<[...]>
        match = _re.search(r"tool_name=<([^>]+)>.*?available_tools=<\[([^\]]+)\]>", text)
        if match:
            called = match.group(1)
            available = [t.strip().strip("'\"") for t in match.group(2).split(",")]
            correct = _find_closest_tool(called, available)
            analysis.tool_errors.append(ToolError(
                task_id=task_id,
                error_type="hallucinated_name",
                tool_called=called,
                available_tools=available,
                correct_tool=correct,
                detail=text[:300],
            ))
            return

        # Pattern 2: generic tool not found
        if "tool not found" in text.lower() or "unknown tool" in text.lower():
            for tc in step.get("tool_calls", []):
                analysis.tool_errors.append(ToolError(
                    task_id=task_id,
                    error_type="tool_not_found",
                    tool_called=tc.get("tool", "unknown"),
                    detail=text[:300],
                ))

        # Pattern 3: parameter / validation errors
        param_keywords = [
            "invalid parameter", "missing required", "schema validation",
            "unexpected keyword", "type error", "validation error",
            "invalid value", "required field",
        ]
        if any(kw in text.lower() for kw in param_keywords):
            for tc in step.get("tool_calls", []):
                analysis.param_errors.append({
                    "task_id": task_id,
                    "tool": tc.get("tool", "unknown"),
                    "input": tc.get("input", {}),
                    "error": text[:500],
                })
                analysis.tool_errors.append(ToolError(
                    task_id=task_id,
                    error_type="wrong_params",
                    tool_called=tc.get("tool", "unknown"),
                    detail=text[:500],
                ))

        # Pattern 4: tool execution errors
        error_keywords = ["error calling tool", "tool execution failed", "api error"]
        if any(kw in text.lower() for kw in error_keywords):
            for tc in step.get("tool_calls", []):
                analysis.tool_errors.append(ToolError(
                    task_id=task_id,
                    error_type="tool_execution_error",
                    tool_called=tc.get("tool", "unknown"),
                    detail=text[:300],
                ))


class McpAutoCorrector:
    """Writes tool-name correction skills and param error memory entries."""

    def apply(
        self,
        workspace: Any,
        analysis: Any,
        accumulated_state: dict[str, Any],
    ) -> int:
        fixes = 0
        name_corrections = accumulated_state.get("name_corrections", {})

        if name_corrections:
            fixes += self._write_name_correction_skill(workspace, name_corrections)

        if analysis.param_errors:
            fixes += self._write_param_memory(workspace, analysis)

        return fixes

    def _write_name_correction_skill(
        self, workspace: Any, corrections: dict[str, str]
    ) -> int:
        if not corrections:
            return 0

        lines = [
            "---",
            "name: tool-name-corrections",
            "description: Maps commonly hallucinated tool names to correct names",
            "---",
            "",
            "# Tool Name Corrections",
            "",
            "Use EXACT tool names. Common mistakes and their corrections:",
            "",
            "| Wrong Name | Correct Name |",
            "|------------|-------------|",
        ]
        for wrong, correct in sorted(corrections.items()):
            lines.append(f"| `{wrong}` | `{correct}` |")

        lines.extend([
            "",
            "Always verify tool names against the available tool list before calling.",
        ])

        content = "\n".join(lines) + "\n"
        skill_path = Path(workspace.root) / "skills" / "tool-name-corrections" / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(content)

        logger.info("Wrote tool-name-corrections skill (%d mappings)", len(corrections))
        return 1

    def _write_param_memory(self, workspace: Any, analysis: Any) -> int:
        if not analysis.param_errors:
            return 0

        memory_dir = Path(workspace.root) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_file = memory_dir / "tool_param_errors.jsonl"

        existing_keys: set[str] = set()
        if memory_file.exists():
            for line in memory_file.read_text().splitlines():
                if line.strip():
                    try:
                        entry = json.loads(line)
                        existing_keys.add(entry.get("tool", ""))
                    except json.JSONDecodeError:
                        pass

        new_entries = 0
        with open(memory_file, "a") as f:
            for pe in analysis.param_errors:
                tool = pe.get("tool", "")
                if tool and tool not in existing_keys:
                    entry = {
                        "tool": tool,
                        "error": pe.get("error", "")[:300],
                        "type": "param_error",
                    }
                    f.write(json.dumps(entry) + "\n")
                    existing_keys.add(tool)
                    new_entries += 1

        if new_entries:
            logger.info("Added %d param error entries to memory", new_entries)
        return new_entries


# ── Utility ──────────────────────────────────────────────────


def _find_closest_tool(called: str, available: list[str]) -> str:
    """Find the most likely intended tool from the available list."""
    called_norm = called.lower().replace("-", "").replace("_", "")

    best_match = ""
    best_score = 0.0

    for tool in available:
        tool_norm = tool.lower().replace("-", "").replace("_", "")

        if called_norm in tool_norm or tool_norm in called_norm:
            score = len(set(called_norm) & set(tool_norm)) / max(
                len(called_norm), len(tool_norm), 1
            )
            if score > best_score:
                best_score = score
                best_match = tool

        prefix_len = 0
        for a, b in zip(called_norm, tool_norm):
            if a == b:
                prefix_len += 1
            else:
                break
        score = prefix_len / max(len(called_norm), len(tool_norm), 1)
        if score > best_score:
            best_score = score
            best_match = tool

    return best_match
