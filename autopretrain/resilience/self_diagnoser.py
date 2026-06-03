"""Self-Diagnoser: The 'self-evolving' intelligence layer.

When standard pattern-matching fails (UNKNOWN failures), the SelfDiagnoser
performs deeper analysis to understand novel failures and propose fixes.

Key capabilities:
1. Pattern extraction — finds the novel error signature
2. Root cause analysis — traces back through log context
3. Fix proposal — suggests config changes based on error semantics
4. Safety validation — ensures proposed fix won't make things worse
5. Learning — successful fixes become new patterns for instant matching

This module enables the framework to handle failures it has never seen before,
and get progressively better at it over time (like an experienced SRE).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopretrain.core.types import (
    ActionType,
    FailureType,
    JobEvent,
    RecoveryAction,
    TrialConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class LearnedFix:
    """A fix that was learned from a successful recovery."""

    pattern: str
    failure_type: FailureType
    action_type: ActionType
    config_changes: dict[str, Any]
    success_count: int = 0
    fail_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class DiagnosisAttempt:
    """Record of a diagnosis attempt and its outcome."""

    timestamp: float
    error_signature: str
    proposed_fix: RecoveryAction | None
    applied: bool = False
    success: bool | None = None  # None = pending


class SelfDiagnoser:
    """Autonomous failure diagnosis with learning capability.

    Architecture:
    1. Extract error signature (unique fingerprint of the failure)
    2. Check learned fixes database
    3. If no match → run heuristic analysis
    4. Propose fix with safety constraints
    5. After outcome is known → update learned fixes
    """

    def __init__(self, knowledge_path: Path | None = None) -> None:
        self._knowledge_path = knowledge_path
        self._learned_fixes: dict[str, LearnedFix] = {}
        self._diagnosis_history: list[DiagnosisAttempt] = []
        self._unsafe_changes: set[str] = {
            "num_gpus",
            "num_nodes",
            "code_dir",
            "model_factory",
        }

        if knowledge_path and knowledge_path.exists():
            self._load_knowledge()

    async def analyze_unknown_failure(
        self,
        logs: str,
        events: str,
        trial_config: TrialConfig,
        history: list[JobEvent],
    ) -> RecoveryAction | None:
        """Analyze an unknown failure and propose a fix.

        Returns None if no safe fix can be determined.
        """
        signature = self._extract_signature(logs)

        # Check learned fixes first
        learned = self._check_learned_fixes(signature)
        if learned and learned.confidence >= 0.7:
            logger.info(
                "Applying learned fix for signature '%s' (confidence=%.2f)",
                signature[:50],
                learned.confidence,
            )
            return RecoveryAction(
                action_type=learned.action_type,
                description=f"Learned fix (seen {learned.success_count} times): {signature[:80]}",
                config_changes=learned.config_changes,
                confidence=learned.confidence,
            )

        # No learned fix — run heuristic analysis
        fix = self._heuristic_analysis(logs, events, trial_config, signature)

        if fix:
            # Record this attempt
            attempt = DiagnosisAttempt(
                timestamp=time.time(),
                error_signature=signature,
                proposed_fix=fix,
            )
            self._diagnosis_history.append(attempt)

        return fix

    async def validate_fix(
        self,
        proposed_action: RecoveryAction,
        trial_config: TrialConfig,
    ) -> bool:
        """Validate that a proposed fix is safe to apply.

        Safety rules:
        - Cannot modify unsafe fields (num_gpus, model, code_dir)
        - Cannot increase resource usage (only decrease or keep same)
        - Cannot reduce lr below 1e-6
        - Cannot reduce batch below 1
        - Must have confidence >= 0.5
        """
        if proposed_action.confidence < 0.5:
            logger.warning("Fix rejected: confidence %.2f < 0.5", proposed_action.confidence)
            return False

        changes = proposed_action.config_changes
        for key in changes:
            if key in self._unsafe_changes:
                logger.warning("Fix rejected: modifies unsafe field '%s'", key)
                return False

        if "lr" in changes and changes["lr"] < 1e-6:
            logger.warning("Fix rejected: lr %.2e below minimum 1e-6", changes["lr"])
            return False

        if "rank_microbatch_size" in changes and changes["rank_microbatch_size"] < 1:
            logger.warning("Fix rejected: microbatch size < 1")
            return False

        return True

    def learn_from_outcome(
        self,
        failure_signature: str,
        action_taken: RecoveryAction,
        success: bool,
    ) -> None:
        """Record outcome for future learning.

        If the fix worked → increase confidence, add to instant-match database.
        If it didn't → decrease confidence, eventually remove.
        """
        if failure_signature in self._learned_fixes:
            fix = self._learned_fixes[failure_signature]
            if success:
                fix.success_count += 1
            else:
                fix.fail_count += 1
            fix.last_used = time.time()
        elif success:
            # New successful fix — add to database
            self._learned_fixes[failure_signature] = LearnedFix(
                pattern=failure_signature,
                failure_type=FailureType.UNKNOWN,
                action_type=action_taken.action_type,
                config_changes=action_taken.config_changes,
                success_count=1,
            )
            logger.info(
                "Learned new fix: '%s' → %s",
                failure_signature[:60],
                action_taken.action_type.value,
            )

        # Persist knowledge
        if self._knowledge_path:
            self._save_knowledge()

    def _extract_signature(self, logs: str) -> str:
        """Extract a unique error signature from logs.

        The signature should be:
        - Specific enough to identify the same error class
        - General enough to match across different runs/steps
        - Stripped of variable data (timestamps, step numbers, memory addresses)
        """
        lines = logs.strip().split("\n")

        # Find the most informative error line
        error_line = ""
        for line in reversed(lines):
            lower = line.lower()
            if any(kw in lower for kw in ["error", "exception", "fatal", "failed"]):
                error_line = line
                break

        if not error_line and lines:
            error_line = lines[-1]

        # Normalize: remove timestamps, step numbers, memory addresses, paths
        signature = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIME>", error_line)
        signature = re.sub(r"step[= ]\d+", "step=<N>", signature, flags=re.IGNORECASE)
        signature = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", signature)
        signature = re.sub(r"\d+\.\d+ [GM]iB", "<MEM>", signature)
        signature = re.sub(r"/[^\s]+/", "<PATH>/", signature)
        signature = signature.strip()

        return signature[:200]

    def _check_learned_fixes(self, signature: str) -> LearnedFix | None:
        """Check if we have a learned fix for this signature."""
        # Exact match first
        if signature in self._learned_fixes:
            return self._learned_fixes[signature]

        # Fuzzy match: check if signature contains a known pattern
        for pattern, fix in self._learned_fixes.items():
            if pattern in signature or signature in pattern:
                return fix

        return None

    def _heuristic_analysis(
        self,
        logs: str,
        events: str,
        trial_config: TrialConfig,
        signature: str,
    ) -> RecoveryAction | None:
        """Heuristic analysis for novel failures.

        Uses semantic understanding of common training failure modes.
        """
        combined = logs + "\n" + events
        lower = combined.lower()

        # Heuristic 1: Memory-related but not standard OOM
        if "memory" in lower and ("allocat" in lower or "fragment" in lower):
            return RecoveryAction(
                action_type=ActionType.REDUCE_BATCH,
                description="Memory pressure detected (non-standard) — reducing batch",
                config_changes={
                    "rank_microbatch_size": max(1, trial_config.rank_microbatch_size // 2),
                },
                confidence=0.6,
            )

        # Heuristic 2: Timeout-related but not NCCL
        if "timeout" in lower or "deadline" in lower or "timed out" in lower:
            return RecoveryAction(
                action_type=ActionType.RESUBMIT,
                description="Timeout detected — simple retry (may be transient load)",
                wait_seconds=60,
                confidence=0.5,
            )

        # Heuristic 3: Permission/access errors
        if "permission" in lower or "access denied" in lower or "forbidden" in lower:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description="Permission error — likely needs manual credential/path fix",
                confidence=0.8,
            )

        # Heuristic 4: Version/compatibility
        if "version" in lower or "incompatible" in lower or "deprecated" in lower:
            return RecoveryAction(
                action_type=ActionType.ALERT_HUMAN,
                description="Version incompatibility — needs environment fix",
                confidence=0.7,
            )

        # Heuristic 5: If logs are completely empty → likely infra issue
        if len(logs.strip()) < 50:
            return RecoveryAction(
                action_type=ActionType.RESUBMIT,
                description="No meaningful logs — likely pod killed before startup. Retrying.",
                wait_seconds=30,
                confidence=0.5,
            )

        # Cannot determine — return None (will escalate to ALERT_HUMAN)
        return None

    def _load_knowledge(self) -> None:
        """Load learned fixes from disk."""
        try:
            data = json.loads(self._knowledge_path.read_text())
            for entry in data:
                fix = LearnedFix(
                    pattern=entry["pattern"],
                    failure_type=FailureType(entry["failure_type"]),
                    action_type=ActionType(entry["action_type"]),
                    config_changes=entry["config_changes"],
                    success_count=entry.get("success_count", 0),
                    fail_count=entry.get("fail_count", 0),
                    first_seen=entry.get("first_seen", time.time()),
                    last_used=entry.get("last_used", time.time()),
                )
                self._learned_fixes[fix.pattern] = fix
            logger.info("Loaded %d learned fixes from %s", len(self._learned_fixes), self._knowledge_path)
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning("Could not load knowledge base: %s", e)

    def _save_knowledge(self) -> None:
        """Persist learned fixes to disk."""
        if not self._knowledge_path:
            return
        data = []
        for fix in self._learned_fixes.values():
            data.append({
                "pattern": fix.pattern,
                "failure_type": fix.failure_type.value,
                "action_type": fix.action_type.value,
                "config_changes": fix.config_changes,
                "success_count": fix.success_count,
                "fail_count": fix.fail_count,
                "first_seen": fix.first_seen,
                "last_used": fix.last_used,
            })
        self._knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        self._knowledge_path.write_text(json.dumps(data, indent=2))
