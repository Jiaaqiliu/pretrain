"""Adaptive self-evolving algorithm with per-claim feedback and in-situ learning.

Self-contained evolution algorithm that includes:
1. Base trajectory analysis (batch-level error extraction, tool errors, strategy issues)
2. Code execution analysis (usage patterns, missed opportunities)
3. Per-claim feedback integration - analyzes which specific requirements failed
4. Task-stratified evolution - tracks performance by task type
5. Judge feedback mining - extracts patterns from LLM judge justifications
6. Meta-evolution learning - learns from evolution history
7. Graduated scope evolution - surgical changes based on failure analysis

For MCP-Atlas benchmark with rich per-claim feedback.
"""

from .analyzer import (
    AdaptiveAnalyzer,
    ClaimAnalyzer,
    ClaimStats,
    JudgeFeedbackMiner,
    TaskTypeDetector,
)
from .base_analysis import (
    AutoCorrector,
    BatchAnalysis,
    ErrorPatternExtractor,
    McpAutoCorrector,
    McpErrorPatternExtractor,
    analyze_observations,
)
from .code_analysis import CodeExecAnalyzer, CodeExecStats
from .engine import AdaptiveEvolveEngine
from .prompts import AdaptivePromptConfig

__all__ = [
    # Engine
    "AdaptiveEvolveEngine",
    # Adaptive analysis
    "AdaptiveAnalyzer",
    "ClaimAnalyzer",
    "ClaimStats",
    "JudgeFeedbackMiner",
    "TaskTypeDetector",
    # Base analysis
    "BatchAnalysis",
    "analyze_observations",
    "McpErrorPatternExtractor",
    "McpAutoCorrector",
    # Code execution analysis
    "CodeExecAnalyzer",
    "CodeExecStats",
    # Protocols
    "ErrorPatternExtractor",
    "AutoCorrector",
    # Config
    "AdaptivePromptConfig",
]
