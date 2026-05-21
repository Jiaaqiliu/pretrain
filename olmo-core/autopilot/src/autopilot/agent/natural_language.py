"""Natural language interface for the AutoPilot agent.

Allows users to configure and control training campaigns through conversational
commands rather than config files or CLI flags. Parses user intent and translates
to structured actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autopilot.experiment.config_builder import ComputeBudget, ModelSize, TrainingPhase, TrainingTarget
from autopilot.optimization.data_mixing import DataDomain
from autopilot.utils.logging import get_logger

log = get_logger("agent.nl_interface")


@dataclass
class ParsedIntent:
    """Structured representation of a user's natural language request."""

    action: str  # "train", "configure", "status", "stop", "adjust", "data", "deploy"
    model_size: Optional[ModelSize] = None
    target_loss: Optional[float] = None
    target_tokens: Optional[int] = None
    data_config: Optional[Dict[str, Any]] = None
    compute_config: Optional[Dict[str, Any]] = None
    phase: Optional[TrainingPhase] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class DataSpec:
    """User's specification of available and needed data."""

    available_domains: List[DataDomain] = field(default_factory=list)
    desired_domains: List[str] = field(default_factory=list)
    missing_domains: List[str] = field(default_factory=list)
    mixture_preferences: Dict[str, float] = field(default_factory=dict)
    total_target_tokens: Optional[int] = None
    quality_requirements: Optional[str] = None


class NaturalLanguageInterface:
    """Parses natural language commands into structured training configurations.

    Supports commands like:
    - "Train a 7B model on web and code data for 2T tokens"
    - "Use 8 nodes with H100 GPUs, target loss 2.5"
    - "I have 500B tokens of web data and 100B tokens of code"
    - "Add math data to the mixture, we need about 50B tokens"
    - "Stop experiment exp_001 and increase learning rate for the rest"
    - "Show me the current loss curves"
    - "Deploy the training on our SLURM cluster"
    """

    # Model size aliases
    _SIZE_PATTERNS = {
        r"\b(\d+)\s*[Mm]\b": "_match_millions",
        r"\b(\d+)\s*[Bb]\b": "_match_billions",
        r"\btiny\b": ModelSize.TINY,
        r"\bsmall\b": ModelSize.SMALL,
        r"\bmedium\b": ModelSize.MEDIUM,
        r"\blarge\b": ModelSize.LARGE,
        r"\b(xl|extra.?large)\b": ModelSize.XL,
        r"\b(xxl|huge)\b": ModelSize.XXL,
    }

    # Common data domain names
    _KNOWN_DOMAINS = [
        "web", "code", "books", "academic", "math", "science",
        "wikipedia", "arxiv", "github", "stackexchange", "common_crawl",
        "c4", "dolma", "redpajama", "fineweb", "starcoder",
    ]

    def parse(self, user_input: str) -> ParsedIntent:
        """Parse a natural language command into structured intent."""
        text = user_input.lower().strip()

        # Detect primary action
        action = self._detect_action(text)

        intent = ParsedIntent(action=action, raw_text=user_input, confidence=0.7)

        # Extract model size
        intent.model_size = self._extract_model_size(text)

        # Extract compute config
        intent.compute_config = self._extract_compute(text)

        # Extract data config
        intent.data_config = self._extract_data_config(text)

        # Extract target metrics
        intent.target_loss = self._extract_target_loss(text)
        intent.target_tokens = self._extract_token_count(text)

        # Extract training phase
        intent.phase = self._extract_phase(text)

        # Extract specific parameters
        intent.parameters = self._extract_parameters(text)

        return intent

    def parse_data_spec(self, user_input: str) -> DataSpec:
        """Parse a data specification from natural language.

        Handles inputs like:
        - "I have 500B web tokens, 100B code tokens, 50B math tokens"
        - "We need more math and science data, about 200B tokens total"
        - "Mix: 60% web, 25% code, 10% math, 5% academic"
        """
        text = user_input.lower()
        spec = DataSpec()

        # Parse "X tokens of Y" patterns (handles "500B tokens of web" and "100B tokens of code")
        have_patterns = re.finditer(
            r"(\d+(?:\.\d+)?)\s*([btmk])\s*(?:tokens?\s+(?:of\s+)?)(\w+)",
            text,
        )
        for match in have_patterns:
            amount_str, unit, domain = match.groups()
            tokens = self._parse_token_amount(amount_str, unit)
            if domain in self._KNOWN_DOMAINS or len(domain) > 2:
                spec.available_domains.append(
                    DataDomain(name=domain, path=f"data/{domain}", token_count=tokens)
                )

        # Parse mixture percentages
        pct_patterns = re.finditer(r"(\d+)\s*%\s*(\w+)", text)
        for match in pct_patterns:
            pct, domain = match.groups()
            spec.mixture_preferences[domain] = float(pct) / 100.0

        # Parse "need/want/missing X data" patterns
        need_patterns = re.finditer(
            r"(?:need|want|missing|lack|add)\s+(?:more\s+)?(\w+)\s+data",
            text,
        )
        for match in need_patterns:
            domain = match.group(1)
            if domain in self._KNOWN_DOMAINS:
                spec.missing_domains.append(domain)

        # Parse total target
        total_match = re.search(r"(\d+(?:\.\d+)?)\s*([btmk])\s*(?:tokens?)?\s*total", text)
        if total_match:
            spec.total_target_tokens = self._parse_token_amount(
                total_match.group(1), total_match.group(2)
            )

        return spec

    def to_training_target(self, intent: ParsedIntent) -> TrainingTarget:
        """Convert parsed intent to a TrainingTarget."""
        compute_budget = None
        if intent.compute_config:
            compute_budget = ComputeBudget(
                num_nodes=intent.compute_config.get("num_nodes", 1),
                gpus_per_node=intent.compute_config.get("gpus_per_node", 8),
                gpu_type=intent.compute_config.get("gpu_type", "A100-80GB"),
            )

        data_domains = []
        if intent.data_config:
            data_domains = intent.data_config.get("domains", [])

        return TrainingTarget(
            model_size=intent.model_size or ModelSize.MEDIUM,
            phase=intent.phase or TrainingPhase.PRETRAIN,
            target_loss=intent.target_loss,
            target_tokens=intent.target_tokens,
            compute_budget=compute_budget,
            data_domains=data_domains,
        )

    def generate_response(self, intent: ParsedIntent, result: Dict[str, Any]) -> str:
        """Generate a natural language response to the user."""
        if intent.action == "train":
            plan_name = result.get("plan_name", "unknown")
            phases = result.get("num_phases", 0)
            hours = result.get("estimated_hours", 0)
            return (
                f"I've created training plan '{plan_name}' with {phases} phases. "
                f"Estimated compute: {hours:.0f} GPU-hours. "
                f"Ready to start when you confirm."
            )
        elif intent.action == "status":
            return self._format_status_response(result)
        elif intent.action == "stop":
            return f"Stopped experiment {result.get('experiment_id', 'unknown')}."
        elif intent.action == "data":
            return self._format_data_response(result)
        else:
            return f"Action '{intent.action}' processed."

    def _detect_action(self, text: str) -> str:
        action_patterns = {
            "train": r"\b(train|pretrain|pre-train|finetune|fine-tune|start training|launch)\b",
            "configure": r"\b(configure|config|set|use|with)\b",
            "status": r"\b(status|progress|how|show|current|loss|metrics)\b",
            "stop": r"\b(stop|cancel|kill|abort|early.?stop)\b",
            "adjust": r"\b(adjust|change|modify|increase|decrease|reduce)\b",
            "data": r"\b(data|dataset|tokens|corpus|mixture|domain)\b",
            "deploy": r"\b(deploy|setup|install|cluster|environment)\b",
        }

        for action, pattern in action_patterns.items():
            if re.search(pattern, text):
                return action
        return "configure"

    def _extract_model_size(self, text: str) -> Optional[ModelSize]:
        # Check for explicit parameter counts
        billions_match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*(?:param|model)?", text)
        if billions_match:
            b = float(billions_match.group(1))
            if b < 0.5:
                return ModelSize.TINY
            elif b < 0.5:
                return ModelSize.SMALL
            elif b < 3:
                return ModelSize.MEDIUM
            elif b < 10:
                return ModelSize.LARGE
            elif b < 20:
                return ModelSize.XL
            else:
                return ModelSize.XXL

        millions_match = re.search(r"(\d+)\s*[Mm](?:illion)?\s*(?:param|model)?", text)
        if millions_match:
            m = int(millions_match.group(1))
            if m < 100:
                return ModelSize.TINY
            elif m < 500:
                return ModelSize.SMALL
            else:
                return ModelSize.MEDIUM

        # Check named sizes
        for pattern, size in [
            (r"\btiny\b", ModelSize.TINY),
            (r"\bsmall\b", ModelSize.SMALL),
            (r"\bmedium\b", ModelSize.MEDIUM),
            (r"\blarge\b", ModelSize.LARGE),
        ]:
            if re.search(pattern, text):
                return size

        return None

    def _extract_compute(self, text: str) -> Optional[Dict[str, Any]]:
        config = {}

        nodes_match = re.search(r"(\d+)\s*nodes?", text)
        if nodes_match:
            config["num_nodes"] = int(nodes_match.group(1))

        gpus_match = re.search(r"(\d+)\s*gpus?(?:\s*per\s*node)?", text)
        if gpus_match:
            config["gpus_per_node"] = int(gpus_match.group(1))

        for gpu_type in ["H100", "H200", "A100", "A10G", "V100"]:
            if gpu_type.lower() in text:
                config["gpu_type"] = gpu_type
                break

        return config if config else None

    def _extract_data_config(self, text: str) -> Optional[Dict[str, Any]]:
        domains = [d for d in self._KNOWN_DOMAINS if d in text]
        if domains:
            return {"domains": domains}
        return None

    def _extract_target_loss(self, text: str) -> Optional[float]:
        match = re.search(r"(?:target|goal)\s*(?:loss)?\s*(?:of\s+|:\s*|=\s*)?([\d.]+)", text)
        if match:
            val = float(match.group(1))
            if 0.5 < val < 10:  # reasonable loss range
                return val
        return None

    def _extract_token_count(self, text: str) -> Optional[int]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*([TtBbMm])\s*tokens?", text)
        if match:
            return self._parse_token_amount(match.group(1), match.group(2))
        return None

    def _extract_phase(self, text: str) -> Optional[TrainingPhase]:
        if re.search(r"\b(pretrain|pre-train|pretraining)\b", text):
            return TrainingPhase.PRETRAIN
        elif re.search(r"\b(finetune|fine-tune|sft|instruction)\b", text):
            return TrainingPhase.SFT
        elif re.search(r"\b(rlhf|reinforcement|reward|dpo)\b", text):
            return TrainingPhase.RLHF
        elif re.search(r"\b(midtrain|mid-train|continued|domain.?adapt)\b", text):
            return TrainingPhase.MIDTRAIN
        return None

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        params = {}

        lr_match = re.search(r"(?:lr|learning.?rate)\s*(?:of\s+|:\s*|=\s*|to\s+)?([\d.e\-+]+)", text)
        if lr_match:
            params["learning_rate"] = float(lr_match.group(1))

        bs_match = re.search(r"(?:batch.?size)\s*(?:of\s+|:\s*|=\s*)?(\d+)", text)
        if bs_match:
            params["batch_size"] = int(bs_match.group(1))

        return params

    def _parse_token_amount(self, amount_str: str, unit: str) -> int:
        amount = float(amount_str)
        unit = unit.upper()
        multipliers = {"T": int(1e12), "B": int(1e9), "M": int(1e6), "K": int(1e3)}
        return int(amount * multipliers.get(unit, 1))

    def _format_status_response(self, result: Dict[str, Any]) -> str:
        status = result.get("status", "unknown")
        active = result.get("active_experiments", 0)
        return f"Campaign status: {status}. {active} experiments running."

    def _format_data_response(self, result: Dict[str, Any]) -> str:
        available = result.get("available_domains", [])
        missing = result.get("missing_domains", [])
        response = f"Data configured: {', '.join(available) if available else 'none'}."
        if missing:
            response += f" Missing: {', '.join(missing)}. I'll look for sources to fill these gaps."
        return response
