"""Multi-layer failure diagnoser.

Diagnosis proceeds through layers of increasing specificity:
1. K8s pod status + exit code (coarse: infra vs training failure)
2. Log regex patterns (specific: OOM, NCCL, data error, etc.)
3. K8s events (node conditions, eviction, scheduling failures)
4. Correlation analysis (multiple trials failing simultaneously → cluster issue)
5. Heartbeat analysis (silent hang vs crash)

Each layer can narrow the diagnosis. If all layers fail → UNKNOWN.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from autopretrain.core.types import FailureType, Heartbeat, JobEvent

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Result of failure diagnosis."""

    failure_type: FailureType
    confidence: float  # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)
    error_context: str = ""
    suggested_action: str = ""


# Regex patterns ordered by specificity (most specific first within each type)
PATTERNS: dict[FailureType, list[str]] = {
    FailureType.OOM: [
        r"CUDA out of memory",
        r"torch\.cuda\.OutOfMemoryError",
        r"OutOfMemoryError",
        r"CUDA error: out of memory",
        r"RuntimeError:.*allocat.*memory",
    ],
    FailureType.NCCL_TIMEOUT: [
        r"NCCL WARN.*Timeout",
        r"NCCL timeout",
        r"Watchdog caught collective operation timeout",
        r"ProcessGroupNCCL.*timeout",
        r"NCCL WARN.*Connect.*fail",
    ],
    FailureType.NETWORK_DISCONNECT: [
        r"Connection closed by peer",
        r"Connection reset by peer",
        r"ConnectionRefusedError",
        r"Socket Timeout",
        r"gloo.*Connection.*closed",
        r"Transport endpoint is not connected",
        r"NetworkError",
    ],
    FailureType.DATA_ERROR: [
        r"Token IDs.*outside valid range",
        r"FileNotFoundError",
        r"No data paths found",
        r"Pattern.*did not match any files",
        r"PermissionError.*data",
        r"IsADirectoryError",
        r"Dataset fingerprint does not match",
    ],
    FailureType.PREEMPTION: [
        r"node.*NotReady",
        r"Evicted",
        r"preempt",
        r"TerminationGracePeriod",
        r"OOMKilled",  # K8s OOMKill (different from CUDA OOM)
    ],
    FailureType.RESOURCE_UNAVAILABLE: [
        r"Insufficient nvidia\.com/gpu",
        r"Insufficient memory",
        r"Unschedulable",
        r"FailedScheduling",
        r"nodes are available",
    ],
    FailureType.NAN_DETECTED: [
        r"loss.*nan",
        r"NaN.*detected",
        r"grad.*nan",
        r"inf.*detected",
    ],
}

# Code bugs — separate because they are never retryable
CODE_BUG_PATTERNS: list[str] = [
    r"ImportError",
    r"ModuleNotFoundError",
    r"SyntaxError",
    r"NameError",
    r"AttributeError:.*has no attribute",
    r"TypeError:.*argument",
    r"IndentationError",
    r"KeyError:(?!.*metric)",  # KeyError but not for metrics
]

# Non-retryable patterns that should immediately alert human
NON_RETRYABLE_PATTERNS: list[str] = [
    r"AssertionError:.*config",
    r"ValueError:.*must be",
    r"pip.*Could not find a version",
]


class MultiLayerDiagnoser:
    """Diagnoses failures through multiple signal layers."""

    def __init__(self) -> None:
        self._learned_patterns: dict[str, FailureType] = {}

    def diagnose(
        self,
        logs: str,
        events: str = "",
        heartbeat: Heartbeat | None = None,
        history: list[JobEvent] | None = None,
        concurrent_failures: int = 0,
    ) -> DiagnosisResult:
        """Run multi-layer diagnosis.

        Args:
            logs: Pod logs (last N lines)
            events: K8s events related to the job
            heartbeat: Last heartbeat signal (if available)
            history: Previous events for this trial
            concurrent_failures: How many other trials failed simultaneously
        """
        combined = logs + "\n" + events
        evidence: list[str] = []

        # Layer 1: Check for silent hang (heartbeat dead but pod "running")
        if heartbeat and heartbeat.is_dead:
            evidence.append(f"Heartbeat dead: last seen {heartbeat.age_seconds:.0f}s ago at step {heartbeat.step}")
            return DiagnosisResult(
                failure_type=FailureType.SILENT_HANG,
                confidence=0.9,
                evidence=evidence,
                error_context=f"Process unresponsive since step {heartbeat.step}",
            )

        # Layer 2: Check for code bugs (non-retryable, check early)
        for pattern in CODE_BUG_PATTERNS:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                evidence.append(f"Code bug pattern: {match.group()}")
                return DiagnosisResult(
                    failure_type=FailureType.CODE_BUG,
                    confidence=0.95,
                    evidence=evidence,
                    error_context=self._extract_error_context(logs),
                )

        # Layer 3: Check known failure patterns
        for failure_type, patterns in PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, combined, re.IGNORECASE)
                if match:
                    evidence.append(f"Pattern match [{failure_type.value}]: {match.group()}")
                    return DiagnosisResult(
                        failure_type=failure_type,
                        confidence=0.85,
                        evidence=evidence,
                        error_context=self._extract_error_context(logs),
                    )

        # Layer 4: Check non-retryable patterns
        for pattern in NON_RETRYABLE_PATTERNS:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                evidence.append(f"Non-retryable: {match.group()}")
                return DiagnosisResult(
                    failure_type=FailureType.CODE_BUG,
                    confidence=0.8,
                    evidence=evidence,
                    error_context=self._extract_error_context(logs),
                )

        # Layer 5: Correlation analysis
        if concurrent_failures >= 2:
            evidence.append(f"{concurrent_failures} trials failed simultaneously → cluster-wide issue")
            return DiagnosisResult(
                failure_type=FailureType.NETWORK_DISCONNECT,
                confidence=0.7,
                evidence=evidence,
                error_context="Cluster-wide failure detected",
                suggested_action="Wait for cluster recovery before retrying",
            )

        # Layer 6: Check learned patterns
        for pattern, failure_type in self._learned_patterns.items():
            if re.search(pattern, combined, re.IGNORECASE):
                evidence.append(f"Learned pattern: {pattern}")
                return DiagnosisResult(
                    failure_type=failure_type,
                    confidence=0.75,
                    evidence=evidence,
                    error_context=self._extract_error_context(logs),
                )

        # Fallback: UNKNOWN
        evidence.append("No known pattern matched")
        return DiagnosisResult(
            failure_type=FailureType.UNKNOWN,
            confidence=0.3,
            evidence=evidence,
            error_context=self._extract_error_context(logs),
        )

    def learn_pattern(self, pattern: str, failure_type: FailureType) -> None:
        """Add a new learned pattern from successful diagnosis."""
        self._learned_patterns[pattern] = failure_type
        logger.info("Learned new failure pattern: '%s' → %s", pattern, failure_type.value)

    def _extract_error_context(self, logs: str, max_lines: int = 30) -> str:
        """Extract the most relevant error context from logs."""
        lines = logs.split("\n")

        # Find the first error/traceback line
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in ["error", "traceback", "exception", "fatal", "critical"]):
                start = max(0, i - 3)
                end = min(len(lines), i + max_lines)
                return "\n".join(lines[start:end])

        # No explicit error found — return last N lines
        return "\n".join(lines[-max_lines:])
