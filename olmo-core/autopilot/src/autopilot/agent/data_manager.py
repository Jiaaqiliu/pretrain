"""Autonomous data management agent.

Handles:
- Discovering available data sources
- Assessing data gaps and suggesting/acquiring missing data
- Managing data mixture and curriculum
- Integrating with known open-source datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from autopilot.optimization.data_mixing import DataDomain, DataMixingOptimizer, MixtureWeights
from autopilot.utils.logging import get_logger

log = get_logger("agent.data_manager")


@dataclass
class DataSource:
    """A known open-source data source."""

    name: str
    description: str
    url: str
    estimated_tokens: int
    domains: List[str]
    quality_tier: str  # "high", "medium", "low"
    license: str
    format: str  # "parquet", "jsonl", "npy", "arrow"
    tokenized: bool = False


# Registry of well-known open-source pre-training data sources
KNOWN_DATA_SOURCES = [
    DataSource(
        name="FineWeb-Edu",
        description="High-quality educational web content filtered by classifiers",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        estimated_tokens=int(1.3e12),
        domains=["web", "academic"],
        quality_tier="high",
        license="ODC-By",
        format="parquet",
    ),
    DataSource(
        name="FineWeb",
        description="Large-scale deduplicated web data from CommonCrawl",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        estimated_tokens=int(15e12),
        domains=["web"],
        quality_tier="medium",
        license="ODC-By",
        format="parquet",
    ),
    DataSource(
        name="StarCoder",
        description="Code from GitHub with permissive licenses",
        url="https://huggingface.co/datasets/bigcode/starcoderdata",
        estimated_tokens=int(250e9),
        domains=["code"],
        quality_tier="high",
        license="Apache-2.0",
        format="parquet",
    ),
    DataSource(
        name="The Stack v2",
        description="Large-scale code dataset from Software Heritage",
        url="https://huggingface.co/datasets/bigcode/the-stack-v2",
        estimated_tokens=int(900e9),
        domains=["code"],
        quality_tier="high",
        license="various",
        format="parquet",
    ),
    DataSource(
        name="RedPajama-v2",
        description="Curated multi-source pre-training dataset",
        url="https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2",
        estimated_tokens=int(30e12),
        domains=["web", "books", "academic", "code"],
        quality_tier="medium",
        license="Apache-2.0",
        format="jsonl",
    ),
    DataSource(
        name="Dolma",
        description="AI2's open corpus for language model pre-training",
        url="https://huggingface.co/datasets/allenai/dolma",
        estimated_tokens=int(3e12),
        domains=["web", "academic", "code", "books", "wikipedia"],
        quality_tier="high",
        license="ODC-By/AI2 ImpACT",
        format="jsonl",
    ),
    DataSource(
        name="OpenWebMath",
        description="High-quality mathematical web content",
        url="https://huggingface.co/datasets/open-web-math/open-web-math",
        estimated_tokens=int(14e9),
        domains=["math"],
        quality_tier="high",
        license="ODC-By",
        format="jsonl",
    ),
    DataSource(
        name="proof-pile-2",
        description="Mathematical text from arXiv, math web, and code",
        url="https://huggingface.co/datasets/EleutherAI/proof-pile-2",
        estimated_tokens=int(55e9),
        domains=["math", "academic"],
        quality_tier="high",
        license="MIT",
        format="jsonl",
    ),
    DataSource(
        name="peS2o",
        description="Semantic Scholar Open Research Corpus",
        url="https://huggingface.co/datasets/allenai/peS2o",
        estimated_tokens=int(40e9),
        domains=["academic", "science"],
        quality_tier="high",
        license="ODC-By",
        format="jsonl",
    ),
    DataSource(
        name="Wikipedia",
        description="Full Wikipedia dumps in multiple languages",
        url="https://huggingface.co/datasets/wikimedia/wikipedia",
        estimated_tokens=int(4e9),
        domains=["wikipedia"],
        quality_tier="high",
        license="CC-BY-SA",
        format="parquet",
    ),
    DataSource(
        name="SlimPajama",
        description="Cleaned, deduplicated RedPajama variant",
        url="https://huggingface.co/datasets/cerebras/SlimPajama-627B",
        estimated_tokens=int(627e9),
        domains=["web", "code", "books", "academic", "wikipedia"],
        quality_tier="medium",
        license="Apache-2.0",
        format="jsonl",
    ),
    DataSource(
        name="DCLM-Baseline",
        description="DataComp-LM curated dataset with model-based filtering",
        url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
        estimated_tokens=int(2.6e12),
        domains=["web"],
        quality_tier="high",
        license="CC-BY-4.0",
        format="jsonl",
    ),
]


@dataclass
class DataGapAnalysis:
    """Analysis of what data is available vs. what is needed."""

    available: Dict[str, int]  # domain -> token count
    required: Dict[str, int]  # domain -> token count needed
    gaps: Dict[str, int]  # domain -> tokens missing
    suggestions: List[DataSource]  # sources to fill gaps
    total_available_tokens: int = 0
    total_required_tokens: int = 0


class DataManagerAgent:
    """Manages training data end-to-end.

    Capabilities:
    1. Discover what data is available (scan paths, check HF datasets)
    2. Analyze gaps between available and required data
    3. Suggest sources to fill gaps from known open-source datasets
    4. Configure optimal data mixture
    5. Set up data pipelines (download, tokenize, prepare)
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        known_sources: Optional[List[DataSource]] = None,
    ):
        self._data_root = Path(data_root) if data_root else None
        self._known_sources = known_sources or KNOWN_DATA_SOURCES
        self._available_domains: Dict[str, DataDomain] = {}
        self._mixer: Optional[DataMixingOptimizer] = None

    @property
    def available_domains(self) -> List[DataDomain]:
        return list(self._available_domains.values())

    def register_data(
        self, name: str, path: str, token_count: int, quality_score: Optional[float] = None
    ) -> None:
        """Register a data source that the user has available."""
        domain = DataDomain(
            name=name, path=path, token_count=token_count, quality_score=quality_score
        )
        self._available_domains[name] = domain
        log.info(f"Registered data domain '{name}': {token_count/1e9:.1f}B tokens at {path}")

    def analyze_gaps(
        self,
        desired_domains: List[str],
        total_target_tokens: int,
        mixture_weights: Optional[Dict[str, float]] = None,
    ) -> DataGapAnalysis:
        """Analyze the gap between available and desired data.

        Args:
            desired_domains: What domains the user wants
            total_target_tokens: How many total tokens are needed
            mixture_weights: Desired proportions per domain (optional)
        """
        # Default to uniform if no weights specified
        if mixture_weights is None:
            n = len(desired_domains)
            mixture_weights = {d: 1.0 / n for d in desired_domains}

        # Normalize weights
        total_weight = sum(mixture_weights.values())
        mixture_weights = {k: v / total_weight for k, v in mixture_weights.items()}

        # Compute requirements
        required = {d: int(total_target_tokens * mixture_weights.get(d, 0)) for d in desired_domains}
        available = {d: self._available_domains[d].token_count for d in desired_domains if d in self._available_domains}

        # Find gaps
        gaps = {}
        for domain in desired_domains:
            have = available.get(domain, 0)
            need = required.get(domain, 0)
            if need > have:
                gaps[domain] = need - have

        # Find suggestions for gaps
        suggestions = self._suggest_sources(gaps)

        return DataGapAnalysis(
            available=available,
            required=required,
            gaps=gaps,
            suggestions=suggestions,
            total_available_tokens=sum(available.values()),
            total_required_tokens=total_target_tokens,
        )

    def suggest_sources(self, domains: List[str]) -> List[DataSource]:
        """Suggest open-source data sources for given domains."""
        return self._suggest_sources({d: 0 for d in domains})

    def build_mixture(self) -> Optional[MixtureWeights]:
        """Build optimal data mixture from registered domains."""
        if not self._available_domains:
            return None

        if self._mixer is None:
            self._mixer = DataMixingOptimizer(list(self._available_domains.values()))

        # Start with quality-weighted mixture
        return self._mixer.quality_weighted_mixture()

    def get_data_pipeline_commands(self, source: DataSource, output_dir: str) -> List[str]:
        """Generate commands to download and prepare a data source."""
        commands = []

        if "huggingface.co" in source.url:
            dataset_id = source.url.split("datasets/")[-1]
            commands.append(
                f"huggingface-cli download {dataset_id} --repo-type dataset "
                f"--local-dir {output_dir}/{source.name}"
            )
        else:
            commands.append(f"# Download from: {source.url}")
            commands.append(f"wget -P {output_dir}/{source.name} {source.url}")

        if not source.tokenized:
            commands.append(
                "# Tokenize with your tokenizer:"
            )
            commands.append(
                f"python -m autopilot.tools.tokenize "
                f"--input {output_dir}/{source.name} "
                f"--output {output_dir}/{source.name}_tokenized "
                f"--format {source.format}"
            )

        return commands

    def discover_local_data(self, scan_path: str) -> List[DataDomain]:
        """Scan a directory for available training data."""
        discovered = []
        scan = Path(scan_path)

        if not scan.exists():
            log.warning(f"Scan path does not exist: {scan_path}")
            return discovered

        # Look for common patterns
        patterns = ["*.npy", "*.jsonl", "*.parquet", "*.arrow", "*.bin"]
        for pattern in patterns:
            for f in scan.rglob(pattern):
                # Estimate token count from file size
                size_bytes = f.stat().st_size
                # Rough estimate: 2 bytes per token for npy, 4 chars per token for text
                if f.suffix == ".npy":
                    est_tokens = size_bytes // 2
                else:
                    est_tokens = size_bytes // 4

                domain_name = f.parent.name
                if domain_name not in self._available_domains:
                    domain = DataDomain(
                        name=domain_name,
                        path=str(f.parent),
                        token_count=est_tokens,
                    )
                    self._available_domains[domain_name] = domain
                    discovered.append(domain)
                    log.info(f"Discovered data: {domain_name} (~{est_tokens/1e9:.1f}B tokens)")

        return discovered

    def _suggest_sources(self, gaps: Dict[str, int]) -> List[DataSource]:
        """Find known data sources that can fill the gaps."""
        suggestions = []
        for domain, needed in gaps.items():
            matching_sources = [
                s for s in self._known_sources if domain in s.domains
            ]
            # Sort by quality tier, then by size
            tier_order = {"high": 0, "medium": 1, "low": 2}
            matching_sources.sort(
                key=lambda s: (tier_order.get(s.quality_tier, 3), -s.estimated_tokens)
            )
            for source in matching_sources:
                if source not in suggestions:
                    suggestions.append(source)
        return suggestions
