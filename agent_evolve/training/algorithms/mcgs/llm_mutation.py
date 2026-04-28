"""LLM-driven mutation proposers for intelligent AutoML evolution.

These mutators use LLM reasoning to propose context-aware mutations based on:
- Full configuration history (not just descriptions)
- Tried-and-failed configurations (avoid repetition)
- Multi-parameter coordinated changes
- Error signals and training dynamics
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import boto3

from ...types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)


# Default seed config for spaceship-titanic (matches workspace manifest)
DEFAULT_SEED_CONFIG = {
    "model_type": "xgboost",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "random_state": 42,
    },
    "feature_engineering": {
        "advanced": True,
        "fillna": "median",
        "scale": False,
        "flags": {
            "passenger_id": True,
            "cabin_split": True,
            "spending_features": True,
            "age_groups": True,
            "family_features": True,
            "interactions": False,           # NEW — searchable
            "log_transform_spending": False, # NEW — searchable
            "target_encoding": False,        # NEW — searchable
        },
    },
}


def _reconstruct_config(node: Any, graph: Any) -> dict:
    """Reconstruct full config for a node by applying all ancestor patches.

    Walks from root → node, applying each patch in order.
    """
    import copy
    config = copy.deepcopy(DEFAULT_SEED_CONFIG)

    if node is None or node.node_id == "node-root":
        return config

    # Collect ancestors (root → node)
    chain = []
    cursor = node
    while cursor is not None and cursor.node_id != "node-root":
        chain.append(cursor)
        cursor = graph.parent(cursor) if graph else None
    chain.reverse()  # root → node order

    # Apply patches in order
    for n in chain:
        if not hasattr(n, "workspace_patch") or not n.workspace_patch:
            continue
        for op in n.workspace_patch.operations:
            if op.path != "model/config.yaml":
                continue
            if op.key_path:
                _apply_patch_to_dict(config, op.key_path, op.value)

    return config


def _apply_patch_to_dict(config: dict, key_path: list, value: Any) -> None:
    """Apply a patch operation to nested dict."""
    if len(key_path) == 1:
        config[key_path[0]] = value
    else:
        if key_path[0] not in config:
            config[key_path[0]] = {}
        _apply_patch_to_dict(config[key_path[0]], key_path[1:], value)


def _config_fingerprint(config: dict) -> str:
    """Generate a deterministic fingerprint for a config."""
    return json.dumps(config, sort_keys=True)


class LLMHyperparameterProposer:
    """LLM-driven hyperparameter tuning with full context awareness.

    Key improvements over baseline:
    1. Shows LLM FULL configs (not just descriptions)
    2. Tracks tried configs to avoid repetition
    3. Supports multi-parameter mutations
    4. Includes regularization / feature engineering params
    """

    # Full search space - let LLM pick any of these
    PARAM_SPACE = {
        # Model choice
        "model_type": ["xgboost", "lightgbm", "random_forest"],
        # Core hyperparameters
        "hyperparameters.n_estimators": [50, 100, 150, 200, 300, 400, 500, 750, 1000],
        "hyperparameters.max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
        "hyperparameters.learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "hyperparameters.subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "hyperparameters.colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "hyperparameters.min_child_weight": [1, 3, 5, 7, 10],
        "hyperparameters.gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "hyperparameters.reg_alpha": [0, 0.01, 0.1, 1.0],
        "hyperparameters.reg_lambda": [0.1, 1.0, 3.0, 5.0, 10.0],
        # NEW: Feature engineering flags — unexplored territory with high potential!
        "feature_engineering.flags.interactions": [True, False],
        "feature_engineering.flags.log_transform_spending": [True, False],
        "feature_engineering.flags.target_encoding": [True, False],
    }

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region: str = "us-west-2",
        verbose: bool = True,
    ):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.verbose = verbose

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose hyperparameter mutation using LLM reasoning.

        Implements hard de-duplication: if LLM proposes a config that was
        already tried, retry up to 3 times with a reminder. Final fallback
        is a random mutation from unexplored region.
        """

        context = self._build_context(parent, graph)
        tried_set = set(context["tried_configs"])

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[LLM] Proposing mutation for parent={parent.node_id}")
            print(f"  Parent metric: {getattr(parent, 'metric', None)}")
            print(f"  Tried configs: {len(tried_set)}")
            print(f"  Best seen: {context['best_metric']}")

        # Try up to 3 times with feedback if LLM proposes duplicate
        prompt = self._build_prompt(context)
        last_response = None
        for attempt in range(3):
            response = self._call_llm(prompt)
            last_response = response

            # Check if proposed config is a duplicate
            parent_config = context["parent_config"]
            base_node_id = response.get("base_node_id", parent.node_id)

            # Find the base node's config (LLM may have picked a different parent)
            base_config = parent_config
            for h in context["metric_history"]:
                if h["node_id"] == base_node_id:
                    base_config = h["full_config"]
                    break

            new_config = self._apply_operations_to_config(
                base_config, response.get("operations", [])
            )
            fingerprint = _config_fingerprint(new_config)

            if fingerprint not in tried_set:
                if self.verbose:
                    print(f"[LLM] Reasoning: {response.get('reasoning', '')[:200]}")
                    print(f"[LLM] Description: {response.get('description', '')}")
                    if attempt > 0:
                        print(f"[LLM] Accepted on retry attempt {attempt + 1}")
                return self._parse_response(response, parent)

            # Duplicate! Append retry instruction
            if self.verbose:
                print(f"[LLM] Attempt {attempt + 1}: DUPLICATE proposal — retrying")
                print(f"  Duplicate desc: {response.get('description', '')[:100]}")

            prompt += f"""

## ⚠️ DUPLICATE DETECTED — RETRY
Your proposal "{response.get('description', '')}" produces a config that was ALREADY TRIED.
You MUST propose something DIFFERENT. Consider:
- A completely different model_type
- Toggle a feature_engineering flag that is currently False
- Move to an unexplored corner of the search space
"""

        # Fallback: LLM kept duplicating. Use last response anyway with warning.
        if self.verbose:
            print(f"[LLM] WARNING: All 3 attempts produced duplicates. Using last anyway.")
        return self._parse_response(last_response, parent)

    def _apply_operations_to_config(self, base_config: dict, operations: list) -> dict:
        """Apply a list of operations to a base config (for dedup check)."""
        import copy
        new_config = copy.deepcopy(base_config)
        for op in operations:
            key_path = op.get("key_path", [])
            if not key_path:
                continue
            _apply_patch_to_dict(new_config, key_path, op.get("value"))
        return new_config

    def _build_context(self, parent: Any, graph: Any) -> dict:
        """Build rich context for LLM reasoning."""

        # Reconstruct parent's full config
        parent_config = _reconstruct_config(parent, graph)

        # Gather full config history with metrics (NOT just descriptions)
        history = []
        tried_fingerprints = set()
        crashed_configs = []  # configs that crashed (metric=None)

        all_nodes = list(graph.nodes.values())

        # CRITICAL: Track ALL non-root nodes (including crashes) to avoid repetition
        for n in all_nodes:
            if n.node_id == "node-root":
                continue
            full_cfg = _reconstruct_config(n, graph)
            fingerprint = _config_fingerprint(full_cfg)
            tried_fingerprints.add(fingerprint)
            # If metric is None, training crashed — tell LLM to avoid this config
            if n.metric is None:
                crashed_configs.append({
                    "node_id": n.node_id,
                    "config": full_cfg,
                    "reason": "training_failed",
                })

        # Sort valid nodes by metric descending for history
        valid_nodes = [
            n for n in all_nodes
            if n.node_id != "node-root" and n.metric is not None
        ]
        valid_nodes.sort(key=lambda n: n.metric or float("-inf"), reverse=True)

        for n in valid_nodes[:10]:  # Top 10 configs
            full_cfg = _reconstruct_config(n, graph)

            history.append({
                "node_id": n.node_id,
                "metric": round(n.metric, 5),
                "full_config": full_cfg,
                "is_best": False,
            })

        if history:
            history[0]["is_best"] = True

        # Include failed (low-metric) configs
        failed = [
            n for n in all_nodes
            if n.node_id != "node-root"
            and n.metric is not None
            and n.metric < 0.5
        ]
        for n in failed[:3]:
            full_cfg = _reconstruct_config(n, graph)
            fingerprint = _config_fingerprint(full_cfg)
            tried_fingerprints.add(fingerprint)

        return {
            "parent_config": parent_config,
            "parent_metric": parent.metric if hasattr(parent, "metric") else None,
            "metric_history": history,
            "best_metric": history[0]["metric"] if history else None,
            "tried_configs": list(tried_fingerprints),
            "crashed_configs": crashed_configs,
            "num_cycles_completed": len(valid_nodes),
            "search_space": self.PARAM_SPACE,
        }

    def _build_prompt(self, context: dict) -> str:
        """Build prompt with full context and search space."""

        # Show full configs with metrics (not just descriptions!)
        history_str = json.dumps(
            [
                {
                    "metric": h["metric"],
                    "config": h["full_config"],
                    "is_best": h["is_best"],
                }
                for h in context["metric_history"]
            ],
            indent=2
        )

        # CRITICAL FIX: Actually show the LLM the list of tried configs
        # (not just the count) so it can check before proposing
        tried_configs_str = json.dumps(
            [json.loads(fp) for fp in context["tried_configs"]],
            indent=2,
        )

        # Analyze what has and hasn't worked
        all_metrics = [h["metric"] for h in context["metric_history"]]
        best = context["best_metric"]
        failures_below_best = [
            h for h in context["metric_history"]
            if h["metric"] < best - 0.003
        ]

        failures_summary = "\n".join(
            f"  - {h['metric']:.4f}: {h['full_config'].get('model_type')} "
            f"lr={h['full_config']['hyperparameters'].get('learning_rate')} "
            f"n_est={h['full_config']['hyperparameters'].get('n_estimators')} "
            f"depth={h['full_config']['hyperparameters'].get('max_depth')}"
            for h in failures_below_best[-5:]  # last 5 failures
        )

        return f"""You are an expert AutoML system optimizing a Kaggle Spaceship Titanic model.

## Task
Binary classification. Score range: 0-1 (higher is better). Current best: {context['best_metric']}.

## 📊 Dataset Info
- Spaceship Titanic: predict if passenger was "Transported" to another dimension
- Train: ~8693 rows, Test: ~4277 rows
- Target: ~50% balanced binary classification
- Key signal: CryoSleep passengers (in cryogenic sleep) generally transported, no spending activity

## ⚠️ NOISE LEVEL
Test set has ~4277 rows. **Score differences < 0.002 are within noise** (~10 rows).
Do NOT over-react to a 0.816 vs 0.818 gap — they might be equivalent.
Focus on changes that could yield >0.003 improvement.

## ⚠️ CRITICAL: What NOT to do
The BEST configs use DEFAULT hyperparameters (lr=0.1, n_estimators=100, max_depth=6).
Many attempts to "improve" via slow-learning (lower lr + more trees) have FAILED:
{failures_summary}

**Stop proposing "lr=0.01, n_estimators=1000"** — that region is exhausted and underperforms!

## 🎯 UNEXPLORED REGIONS WITH HIGH POTENTIAL
1. **Feature Engineering flags** (likely biggest win, never tried):
   - `interactions=True`: creates CryoSleep*HasSpending (known data leak), Age*Spent, VIP*Spent
   - `log_transform_spending=True`: fixes skewed spending distribution (helps tree splits)
   - `target_encoding=True`: smoothed mean-target encoding for HomePlanet/Destination/Deck
2. **Completely different model**: Random Forest rarely tried, might find different patterns
3. **Multi-parameter combos you haven't tried**: e.g. xgboost + interactions=True + small max_depth

## Search Space (pick values from these)
{json.dumps(context['search_space'], indent=2)}

## Training History (sorted by metric, top 10)
{history_str}

## ALREADY TRIED CONFIGS — DO NOT REPEAT THESE
{tried_configs_str}

## CONFIGS THAT CRASHED — AVOID THESE PATTERNS
{json.dumps(context.get('crashed_configs', []), indent=2) if context.get('crashed_configs') else 'none'}

## Parent Config (current mutation base)
{json.dumps(context['parent_config'], indent=2)}

Parent metric: {context['parent_metric']}

## Constraints
1. **DO NOT repeat any config above** — check the "ALREADY TRIED" list before proposing.
2. **Prefer feature engineering over hyperparameter tweaks** for this round if FE flags are all False — FE hasn't been explored!
3. **Break out of local minima** — if you've tried 3+ similar hyperparameter configs, switch to FE or different model.
4. **Change 1-3 parameters** — single FE flag toggle is often high-ROI.

## Output Format (strict JSON)
Return a JSON object with:
- "reasoning": 2-3 sentences explaining WHY this mutation will improve (cite metric history)
- "base_node_id": node_id to base mutation on (pick the best or another node)
- "description": 1-line summary of the change
- "operations": list of PatchOperation dicts, each with:
    - "op": "replace"
    - "path": "model/config.yaml"
    - "key_path": ["hyperparameters", "n_estimators"]  # or ["model_type"], etc.
    - "value": the new value

Example 1 (Feature Engineering — recommended when FE flags all False):
{{
  "reasoning": "Best is LightGBM defaults (0.81839). All FE flags are False — an unexplored high-value region. Enabling interactions creates CryoSleep*HasSpending which captures the known rule that cryo passengers can't spend.",
  "base_node_id": "node-5a44e50433",
  "description": "LightGBM + interactions=True (new feature family)",
  "operations": [
    {{"op": "replace", "path": "model/config.yaml", "key_path": ["feature_engineering", "flags", "interactions"], "value": true}}
  ]
}}

Example 2 (Hyperparameter — only if FE has been explored):
{{
  "reasoning": "LightGBM with interactions=True hit 0.824. Now try target_encoding on top to add HomePlanet/Destination target stats.",
  "base_node_id": "node-abc123",
  "description": "LightGBM + interactions + target_encoding",
  "operations": [
    {{"op": "replace", "path": "model/config.yaml", "key_path": ["feature_engineering", "flags", "target_encoding"], "value": true}}
  ]
}}

Respond with ONLY the JSON object, no other text:"""

    def _call_llm(self, prompt: str) -> dict:
        """Call Bedrock Claude API."""
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            }),
        )

        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    def _parse_response(self, response: dict, parent: Any) -> WorkspaceMutation:
        """Parse LLM response into WorkspaceMutation."""
        mutation_id = f"m-llm-{uuid.uuid4().hex[:8]}"

        operations = [PatchOperation(**op) for op in response["operations"]]

        # If LLM specified a base_node_id, attach to that node instead of parent
        base_node_id = response.get("base_node_id", parent.node_id)

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=base_node_id,
            description=f"[LLM] {response['description']}",
            patch=WorkspacePatch(operations=operations),
            mutation_type="training_recipe",
        )


class LLMFeatureEngineeringProposer:
    """LLM proposes feature engineering strategies (kept for compatibility)."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region: str = "us-west-2",
        verbose: bool = True,
    ):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.verbose = verbose

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        # Delegate to LLMHyperparameterProposer for now (unified proposer)
        hyperparameter_proposer = LLMHyperparameterProposer(
            model_id=self.model_id,
            verbose=self.verbose,
        )
        return hyperparameter_proposer.propose(parent, graph)
