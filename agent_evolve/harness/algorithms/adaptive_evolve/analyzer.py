"""Enhanced trajectory analyzer with per-claim feedback and task-type awareness.

Extends code_evolve analyzer with:
1. Per-claim analysis to identify which specific requirements fail
2. Task type detection and stratified performance tracking
3. Judge feedback mining to extract root cause patterns
4. Claim-type performance matrix for targeted evolution
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .code_analysis import CodeExecAnalyzer, CodeExecStats
from .base_analysis import BatchAnalysis


# ── Claim Type Taxonomy ─────────────────────────────────────────

CLAIM_TYPES = {
    # Information retrieval
    "provide_fact": ["provide", "what is", "get", "return", "show"],
    "calculate": ["difference", "sum", "calculate", "how many", "count"],
    "compare": ["compare", "difference between", "versus", "vs"],
    "aggregate": ["total", "all", "list all", "every"],

    # Entity operations
    "identify_entity": ["identify", "find", "which", "who is"],
    "entity_property": ["status", "date", "name", "owner", "created", "updated"],

    # Multi-step
    "chain": ["then", "after", "using", "next"],
    "conditional": ["if", "when", "where", "in case"],
}

TASK_TYPE_PATTERNS = {
    "single_fact": {
        "keywords": ["what is", "when was", "who is", "where is"],
        "complexity": "low",
        "expected_calls": (1, 3),
    },
    "multi_requirement": {
        "signals": [" and ", " also ", " additionally ", "\n-", "\n•", " plus "],
        "complexity": "medium",
        "expected_calls": (3, 8),
    },
    "search_iteration": {
        "keywords": ["find", "search for", "all", "list", "every"],
        "complexity": "medium-high",
        "expected_calls": (5, 15),
    },
    "comparison": {
        "keywords": ["compare", "difference between", "versus", "vs"],
        "complexity": "medium",
        "expected_calls": (4, 8),
    },
    "action": {
        "keywords": ["create", "update", "delete", "send", "post", "add", "remove"],
        "complexity": "medium",
        "expected_calls": (3, 7),
    },
    "calculation": {
        "keywords": ["calculate", "compute", "sum", "total", "average"],
        "complexity": "low-medium",
        "expected_calls": (2, 5),
    },
}


# ── Data Classes ────────────────────────────────────────────────

@dataclass
class ClaimStats:
    """Statistics for a specific claim type."""
    claim_type: str
    total: int = 0
    fulfilled: int = 0
    partial: int = 0
    failed: int = 0
    examples: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.fulfilled + 0.5 * self.partial) / self.total if self.total > 0 else 0.0

    @property
    def full_fulfill_rate(self) -> float:
        return self.fulfilled / self.total if self.total > 0 else 0.0


@dataclass
class TaskTypeStats:
    """Performance statistics for a task type."""
    task_type: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    scores: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class FailurePattern:
    """A specific failure pattern identified."""
    pattern_name: str
    count: int
    task_ids: list[str]
    description: str
    suggested_fix: str


@dataclass
class AdaptiveAnalysisResult:
    """Complete analysis result with all enhancement layers."""
    # Base analysis
    base_analysis: BatchAnalysis
    code_stats: CodeExecStats

    # Enhanced layers
    claim_stats: dict[str, ClaimStats]
    task_type_stats: dict[str, TaskTypeStats]
    judge_patterns: dict[str, list[dict]]
    failure_patterns: list[FailurePattern]

    # Meta information
    weakest_claim_types: list[tuple[str, float]]  # (type, pass_rate)
    weakest_task_types: list[tuple[str, float]]
    evolution_recommendations: list[str]


# ── Task Type Detector ──────────────────────────────────────────

class TaskTypeDetector:
    """Detect task types from input text."""

    def detect(self, task_input: str) -> dict[str, Any]:
        """Detect task type and characteristics.

        Returns:
            Dict with keys: type, complexity, requirement_count, signals
        """
        task_lower = task_input.lower()

        # Check multi-requirement signals first (high priority)
        req_signals = []
        req_count = 0
        for signal in TASK_TYPE_PATTERNS["multi_requirement"]["signals"]:
            count = task_input.count(signal) if signal.startswith("\n") else task_lower.count(signal)
            if count > 0:
                req_signals.append((signal, count))
                req_count += count

        if req_count >= 2:
            return {
                "type": "multi_requirement",
                "requirement_count": req_count + 1,
                "complexity": "medium",
                "signals": req_signals,
            }

        # Check other patterns
        for task_type, pattern in TASK_TYPE_PATTERNS.items():
            if task_type == "multi_requirement":
                continue

            if "keywords" in pattern:
                matched_kw = [kw for kw in pattern["keywords"] if kw in task_lower]
                if matched_kw:
                    return {
                        "type": task_type,
                        "complexity": pattern["complexity"],
                        "expected_calls": pattern.get("expected_calls"),
                        "matched_keywords": matched_kw,
                    }

        return {
            "type": "unknown",
            "complexity": "medium",
        }


# ── Claim Analyzer ──────────────────────────────────────────────

class ClaimAnalyzer:
    """Analyzes per-claim failures to identify systematic weaknesses."""

    def analyze_claims(self, observations: list[dict[str, Any]]) -> dict[str, ClaimStats]:
        """Build claim-type performance matrix from observations.

        Args:
            observations: List of observation dicts with feedback.raw.per_claim

        Returns:
            Dict mapping claim_type -> ClaimStats
        """
        claim_stats_map: dict[str, ClaimStats] = {}

        for obs in observations:
            feedback_raw = obs.get("feedback", {}).get("raw", {})
            per_claim = feedback_raw.get("per_claim", [])

            for claim_data in per_claim:
                claim_text = claim_data.get("claim", "").lower()
                claim_type = self._classify_claim(claim_text)

                if claim_type not in claim_stats_map:
                    claim_stats_map[claim_type] = ClaimStats(claim_type=claim_type)

                stats = claim_stats_map[claim_type]
                stats.total += 1

                score = claim_data.get("score", 0.0)
                if score >= 1.0:
                    stats.fulfilled += 1
                elif score >= 0.5:
                    stats.partial += 1
                else:
                    stats.failed += 1
                    # Store failed examples (up to 3 per type)
                    if len(stats.examples) < 3:
                        stats.examples.append({
                            "task_id": obs.get("task_id"),
                            "claim": claim_data.get("claim"),
                            "outcome": claim_data.get("outcome", "not_fulfilled"),
                            "justification": claim_data.get("justification", ""),
                        })

        return claim_stats_map

    def _classify_claim(self, claim_text: str) -> str:
        """Classify claim into type based on keywords."""
        for claim_type, keywords in CLAIM_TYPES.items():
            if any(kw in claim_text for kw in keywords):
                return claim_type
        return "other"

    def get_weakest_claim_types(
        self, claim_stats: dict[str, ClaimStats], limit: int = 3
    ) -> list[tuple[str, float]]:
        """Return weakest performing claim types.

        Returns:
            List of (claim_type, pass_rate) tuples, sorted by pass rate
        """
        return sorted(
            [(ct, stats.pass_rate) for ct, stats in claim_stats.items()],
            key=lambda x: x[1]
        )[:limit]


# ── Judge Feedback Miner ────────────────────────────────────────

class JudgeFeedbackMiner:
    """Extract insights from LLM judge justifications."""

    REASON_PATTERNS = {
        "missing_data": [
            "not mention", "missing", "not provide", "does not include",
            "fails to", "omit", "absent"
        ],
        "wrong_format": [
            "format", "structure", "not formatted", "should be", "expected format"
        ],
        "partial_answer": [
            "partial", "incomplete", "only provides", "missing some",
            "not all", "some but not"
        ],
        "wrong_entity": [
            "wrong", "different", "incorrect", "not the", "another",
            "refers to different"
        ],
        "calculation_error": [
            "incorrect calculation", "wrong number", "math error",
            "miscalculated", "should be"
        ],
    }

    def mine_patterns(self, observations: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Extract common failure reasons from judge justifications.

        Args:
            observations: List of observation dicts

        Returns:
            Dict mapping pattern_name -> list of examples
        """
        patterns: dict[str, list[dict]] = {
            pattern: [] for pattern in self.REASON_PATTERNS
        }

        for obs in observations:
            if obs.get("success", True):
                continue

            feedback_raw = obs.get("feedback", {}).get("raw", {})
            per_claim = feedback_raw.get("per_claim", [])

            for claim in per_claim:
                if claim.get("score", 1.0) >= 0.5:
                    continue  # Only analyze failures

                justification = claim.get("justification", "").lower()

                for pattern_name, keywords in self.REASON_PATTERNS.items():
                    if any(kw in justification for kw in keywords):
                        patterns[pattern_name].append({
                            "task_id": obs.get("task_id"),
                            "claim": claim.get("claim"),
                            "justification": claim.get("justification"),
                            "score": claim.get("score", 0.0),
                        })
                        break  # Only count each failure once

        # Remove empty patterns
        return {k: v for k, v in patterns.items() if v}


# ── Task Type Performance Tracker ───────────────────────────────

class TaskTypePerformanceTracker:
    """Track performance by task type over time."""

    def __init__(self):
        self.detector = TaskTypeDetector()
        self.history: dict[str, TaskTypeStats] = {}

    def analyze_batch(self, observations: list[dict[str, Any]]) -> dict[str, TaskTypeStats]:
        """Analyze task type performance for a batch.

        Returns:
            Dict mapping task_type -> TaskTypeStats
        """
        batch_stats: dict[str, TaskTypeStats] = {}

        for obs in observations:
            task_input = obs.get("task_input", obs.get("input", ""))
            task_info = self.detector.detect(task_input)
            task_type = task_info["type"]

            if task_type not in batch_stats:
                batch_stats[task_type] = TaskTypeStats(task_type=task_type)

            stats = batch_stats[task_type]
            stats.total += 1

            success = obs.get("success", False)
            score = obs.get("score", 0.0)

            if success:
                stats.passed += 1
            else:
                stats.failed += 1

            stats.scores.append(score)

        # Calculate average scores
        for stats in batch_stats.values():
            if stats.scores:
                stats.avg_score = sum(stats.scores) / len(stats.scores)

        # Update history (cumulative)
        for task_type, batch_stat in batch_stats.items():
            if task_type not in self.history:
                self.history[task_type] = TaskTypeStats(task_type=task_type)

            hist = self.history[task_type]
            hist.total += batch_stat.total
            hist.passed += batch_stat.passed
            hist.failed += batch_stat.failed
            hist.scores.extend(batch_stat.scores)
            hist.avg_score = sum(hist.scores) / len(hist.scores) if hist.scores else 0.0

        return batch_stats

    def get_weakest_types(self, limit: int = 3) -> list[tuple[str, float]]:
        """Return weakest performing task types from history.

        Returns:
            List of (task_type, pass_rate) tuples
        """
        return sorted(
            [(t, stats.pass_rate) for t, stats in self.history.items()],
            key=lambda x: x[1]
        )[:limit]


# ── Failure Pattern Detector ────────────────────────────────────

class FailurePatternDetector:
    """Detect specific failure patterns for targeted fixes."""

    def detect_patterns(self, observations: list[dict[str, Any]]) -> list[FailurePattern]:
        """Identify failure patterns in the batch.

        Returns:
            List of detected FailurePattern objects
        """
        patterns = []

        # Pattern 1: Multi-requirement misses (score ~0.5)
        multi_req_misses = []
        for obs in observations:
            score = obs.get("score", 1.0)
            if 0.45 <= score <= 0.55:
                task_input = obs.get("task_input", "")
                if " and " in task_input or " also " in task_input:
                    multi_req_misses.append(obs.get("task_id"))

        if len(multi_req_misses) >= 3:
            patterns.append(FailurePattern(
                pattern_name="multi_requirement_miss",
                count=len(multi_req_misses),
                task_ids=multi_req_misses[:5],
                description="Agent fulfills some requirements but misses others (score ~0.5)",
                suggested_fix="Add structured requirement extraction protocol"
            ))

        # Pattern 2: Complete misses (score 0.0)
        complete_misses = []
        for obs in observations:
            score = obs.get("score", 1.0)
            output_len = len(obs.get("output", ""))
            if score == 0.0 and output_len > 100:
                complete_misses.append(obs.get("task_id"))

        if len(complete_misses) >= 2:
            patterns.append(FailurePattern(
                pattern_name="wrong_entity_targeting",
                count=len(complete_misses),
                task_ids=complete_misses[:5],
                description="Agent produces output but scores 0.0 (wrong entity likely)",
                suggested_fix="Add early entity verification checkpoint"
            ))

        # Pattern 3: Near misses (score 0.67-0.73)
        near_misses = []
        for obs in observations:
            score = obs.get("score", 1.0)
            if 0.65 <= score <= 0.75:
                near_misses.append(obs.get("task_id"))

        if len(near_misses) >= 3:
            patterns.append(FailurePattern(
                pattern_name="near_miss",
                count=len(near_misses),
                task_ids=near_misses[:5],
                description="Agent gets most claims right but misses one detail",
                suggested_fix="Strengthen final verification: check EVERY requirement"
            ))

        # Pattern 4: Code execution under-utilization
        code_needed = []
        for obs in observations:
            steps = obs.get("steps", [])
            total_calls = sum(len(step.get("tool_calls", [])) for step in steps)
            code_calls = sum(
                1 for step in steps
                for tc in step.get("tool_calls", [])
                if "execute_code" in tc.get("tool", "")
            )

            if total_calls >= 15 and code_calls == 0 and not obs.get("success", False):
                code_needed.append(obs.get("task_id"))

        if len(code_needed) >= 2:
            patterns.append(FailurePattern(
                pattern_name="missed_code_opportunity",
                count=len(code_needed),
                task_ids=code_needed[:5],
                description="Tasks with 15+ tool calls but no code execution (search/iteration likely)",
                suggested_fix="Lower code execution threshold: use for search tasks at 10+ calls"
            ))

        return patterns


# ── Main Adaptive Analyzer ──────────────────────────────────────

class AdaptiveAnalyzer:
    """Main analyzer combining all enhancement layers."""

    def __init__(self):
        self.claim_analyzer = ClaimAnalyzer()
        self.feedback_miner = JudgeFeedbackMiner()
        self.type_tracker = TaskTypePerformanceTracker()
        self.pattern_detector = FailurePatternDetector()
        self.code_analyzer = CodeExecAnalyzer()

    def analyze(
        self,
        observations: list[dict[str, Any]],
        base_analysis: BatchAnalysis,
        code_stats: CodeExecStats,
    ) -> AdaptiveAnalysisResult:
        """Perform complete adaptive analysis.

        Args:
            observations: List of observation dicts from solve cycles
            base_analysis: Base BatchAnalysis from mcp_evolve
            code_stats: CodeExecStats from code_evolve

        Returns:
            AdaptiveAnalysisResult with all layers of analysis
        """
        # Layer 1: Claim analysis
        claim_stats = self.claim_analyzer.analyze_claims(observations)
        weakest_claims = self.claim_analyzer.get_weakest_claim_types(claim_stats)

        # Layer 2: Task type analysis
        task_type_stats = self.type_tracker.analyze_batch(observations)
        weakest_types = self.type_tracker.get_weakest_types()

        # Layer 3: Judge feedback mining
        judge_patterns = self.feedback_miner.mine_patterns(observations)

        # Layer 4: Failure pattern detection
        failure_patterns = self.pattern_detector.detect_patterns(observations)

        # Layer 5: Generate recommendations
        recommendations = self._generate_recommendations(
            claim_stats, task_type_stats, judge_patterns, failure_patterns
        )

        return AdaptiveAnalysisResult(
            base_analysis=base_analysis,
            code_stats=code_stats,
            claim_stats=claim_stats,
            task_type_stats=task_type_stats,
            judge_patterns=judge_patterns,
            failure_patterns=failure_patterns,
            weakest_claim_types=weakest_claims,
            weakest_task_types=weakest_types,
            evolution_recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        claim_stats: dict[str, ClaimStats],
        task_type_stats: dict[str, TaskTypeStats],
        judge_patterns: dict[str, list[dict]],
        failure_patterns: list[FailurePattern],
    ) -> list[str]:
        """Generate actionable evolution recommendations."""
        recommendations = []

        # From failure patterns
        for pattern in failure_patterns:
            recommendations.append(f"{pattern.pattern_name}: {pattern.suggested_fix}")

        # From claim analysis
        if claim_stats:
            weakest = sorted(claim_stats.items(), key=lambda x: x[1].pass_rate)[:2]
            for claim_type, stats in weakest:
                if stats.pass_rate < 0.6:
                    recommendations.append(
                        f"Weak claim type '{claim_type}' ({stats.pass_rate:.0%}): "
                        f"Create skill or prompt guidance for this type"
                    )

        # From judge patterns
        if judge_patterns:
            top_pattern = max(judge_patterns.items(), key=lambda x: len(x[1]))
            pattern_name, examples = top_pattern
            recommendations.append(
                f"Common judge feedback: '{pattern_name}' ({len(examples)} times) - "
                f"Address this systematically"
            )

        # From task types
        if task_type_stats:
            for task_type, stats in task_type_stats.items():
                if stats.pass_rate < 0.65 and stats.total >= 3:
                    recommendations.append(
                        f"Weak task type '{task_type}' ({stats.pass_rate:.0%}): "
                        f"Create type-specific handling skill"
                    )

        return recommendations
