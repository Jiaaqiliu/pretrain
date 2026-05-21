"""LLM-powered reasoning engine for complex training decisions.

Uses Claude/GPT-4 API to:
- Diagnose complex training failures that rule-based systems can't handle
- Generate training strategy recommendations
- Analyze experiment results and provide natural language insights
- Answer user questions about training progress
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autopilot.utils.logging import get_logger

log = get_logger("agent.llm_reasoning")


@dataclass
class ReasoningContext:
    """Context provided to the LLM for reasoning."""

    experiment_id: str
    current_metrics: Dict[str, float]
    recent_history: List[Dict[str, float]]
    anomalies: List[Dict[str, Any]]
    config: Dict[str, Any]
    previous_decisions: List[Dict[str, Any]]
    user_query: Optional[str] = None


@dataclass
class ReasoningResult:
    """Structured output from LLM reasoning."""

    diagnosis: str
    recommended_actions: List[Dict[str, Any]]
    confidence: float
    explanation: str
    follow_up_questions: List[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are AutoPilot's reasoning engine — an expert in large language model training.

Your role is to analyze training metrics, diagnose issues, and recommend actions.

You have deep knowledge of:
- LLM training dynamics (loss curves, gradient behavior, learning rate schedules)
- Common failure modes (loss spikes, divergence, slow convergence, instabilities)
- Scaling laws (Chinchilla, muTransfer, data mixing laws)
- Best practices from OLMo, Llama, DeepSeek training reports

When analyzing, consider:
1. Is the training progressing normally for this model size and step count?
2. Are there any warning signs of upcoming problems?
3. What specific, actionable changes would improve training?

Always respond with valid JSON matching this schema:
{
    "diagnosis": "Brief description of the situation",
    "recommended_actions": [{"action": "action_type", "params": {...}, "priority": "high/medium/low"}],
    "confidence": 0.0-1.0,
    "explanation": "Detailed reasoning",
    "follow_up_questions": ["Questions to gather more info if needed"]
}
"""


class LLMReasoningEngine:
    """Uses LLM APIs for complex decision-making beyond rule-based logic.

    Supports multiple providers:
    - Anthropic (Claude)
    - OpenAI (GPT-4)
    - Local models via compatible APIs
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("pip install anthropic")
        elif self._provider == "openai":
            try:
                import openai

                kwargs = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise RuntimeError("pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

        return self._client

    def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Submit a reasoning request to the LLM."""
        user_message = self._build_user_message(context)

        try:
            response_text = self._call_llm(user_message)
            return self._parse_response(response_text)
        except Exception as e:
            log.warning(f"LLM reasoning failed: {e}")
            return ReasoningResult(
                diagnosis="LLM reasoning unavailable",
                recommended_actions=[],
                confidence=0.0,
                explanation=f"Error: {e}",
            )

    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ask the LLM a free-form question about training."""
        message = question
        if context:
            message = f"Context:\n{json.dumps(context, indent=2)}\n\nQuestion: {question}"

        try:
            return self._call_llm(message)
        except Exception as e:
            return f"Unable to get LLM response: {e}"

    def diagnose_failure(
        self,
        experiment_id: str,
        error_logs: str,
        config: Dict[str, Any],
        metrics_history: List[Dict[str, float]],
    ) -> ReasoningResult:
        """Diagnose a training failure from logs and metrics."""
        context = ReasoningContext(
            experiment_id=experiment_id,
            current_metrics=metrics_history[-1] if metrics_history else {},
            recent_history=metrics_history[-20:],
            anomalies=[],
            config=config,
            previous_decisions=[],
            user_query=f"The training job failed. Error logs:\n{error_logs[:2000]}",
        )
        return self.reason(context)

    def suggest_next_experiment(
        self,
        completed_experiments: List[Dict[str, Any]],
        current_best: Dict[str, Any],
        remaining_budget_hours: float,
    ) -> ReasoningResult:
        """Suggest what experiment to run next based on history."""
        context = ReasoningContext(
            experiment_id="planning",
            current_metrics=current_best.get("metrics", {}),
            recent_history=[],
            anomalies=[],
            config=current_best.get("config", {}),
            previous_decisions=[],
            user_query=(
                f"We have {remaining_budget_hours:.0f} GPU-hours remaining. "
                f"Completed {len(completed_experiments)} experiments. "
                f"Best result: {current_best}. "
                f"What should we try next?"
            ),
        )
        return self.reason(context)

    def _build_user_message(self, context: ReasoningContext) -> str:
        parts = [f"Experiment: {context.experiment_id}"]

        if context.current_metrics:
            parts.append(f"Current metrics: {json.dumps(context.current_metrics)}")

        if context.recent_history:
            parts.append(f"Recent history ({len(context.recent_history)} steps):")
            for entry in context.recent_history[-5:]:
                parts.append(f"  {json.dumps(entry)}")

        if context.anomalies:
            parts.append(f"Detected anomalies: {json.dumps(context.anomalies)}")

        if context.config:
            parts.append(f"Training config: {json.dumps(context.config, indent=2)}")

        if context.previous_decisions:
            parts.append(f"Previous decisions: {json.dumps(context.previous_decisions[-5:])}")

        if context.user_query:
            parts.append(f"\nUser query: {context.user_query}")

        return "\n\n".join(parts)

    def _call_llm(self, user_message: str) -> str:
        client = self._get_client()

        if self._provider == "anthropic":
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        elif self._provider == "openai":
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self._provider}")

    def _parse_response(self, text: str) -> ReasoningResult:
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            return ReasoningResult(
                diagnosis=data.get("diagnosis", ""),
                recommended_actions=data.get("recommended_actions", []),
                confidence=data.get("confidence", 0.5),
                explanation=data.get("explanation", ""),
                follow_up_questions=data.get("follow_up_questions", []),
            )
        except (json.JSONDecodeError, IndexError):
            return ReasoningResult(
                diagnosis=text[:200],
                recommended_actions=[],
                confidence=0.3,
                explanation=text,
            )
