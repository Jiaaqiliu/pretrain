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
from pathlib import Path
from typing import Any

import boto3

from ...types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)


def _compute_dataset_profile(
    workspace_root: Path,
    train_filename: str = "train.csv",
    target_column: str | None = None,
    id_column: str | None = None,
) -> dict:
    """Compute objective dataset statistics from train.csv.

    Returns an empty dict if the file can't be loaded (profile is optional
    context for the LLM, not required). Uses only observable properties of
    the data — no dataset name, no Kaggle metadata — so the LLM can't use
    prior knowledge of a specific competition.
    """
    try:
        import pandas as pd
        import numpy as np

        train_path = workspace_root / "data" / train_filename
        if not train_path.exists():
            return {}
        df = pd.read_csv(train_path)

        profile: dict = {"n_train_rows": int(len(df))}

        # Target stats
        if target_column and target_column in df.columns:
            y = df[target_column]
            profile["target_column"] = target_column
            profile["target_dtype"] = str(y.dtype)
            if pd.api.types.is_numeric_dtype(y):
                n_unique = int(y.nunique())
                if n_unique <= 20:
                    # Classification-like: show class counts
                    vc = y.value_counts(normalize=True).sort_index()
                    profile["n_classes"] = n_unique
                    profile["target_class_balance"] = {
                        str(k): round(float(v), 4) for k, v in vc.items()
                    }
                else:
                    # Regression-like: range + skew
                    profile["target_mean"] = round(float(y.mean()), 4)
                    profile["target_std"] = round(float(y.std()), 4)
                    profile["target_min"] = round(float(y.min()), 4)
                    profile["target_max"] = round(float(y.max()), 4)
                    try:
                        profile["target_skew"] = round(float(y.skew()), 3)
                    except Exception:
                        pass
            df_features = df.drop(columns=[target_column])
        else:
            df_features = df

        # Drop likely ID column for feature stats
        if id_column and id_column in df_features.columns:
            df_features = df_features.drop(columns=[id_column])

        profile["n_features"] = int(df_features.shape[1])

        # Dtype breakdown
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_features.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        bool_cols = df_features.select_dtypes(include=["bool"]).columns.tolist()
        profile["n_numeric_features"] = len(numeric_cols)
        profile["n_categorical_features"] = len(cat_cols)
        profile["n_boolean_features"] = len(bool_cols)

        # Missing rate
        missing_per_col = df_features.isna().mean()
        profile["missing_rate_overall"] = round(float(missing_per_col.mean()), 4)
        profile["features_with_missing"] = int((missing_per_col > 0).sum())
        top_missing = missing_per_col[missing_per_col > 0].sort_values(ascending=False).head(5)
        if len(top_missing) > 0:
            profile["top_missing_features"] = {
                k: round(float(v), 4) for k, v in top_missing.items()
            }

        # Numeric skew summary (informs log_transform_skewed decisions)
        if numeric_cols:
            skews = df_features[numeric_cols].skew().abs()
            high_skew = (skews > 1.0).sum()
            profile["numeric_high_skew_count"] = int(high_skew)
            profile["numeric_max_skew"] = round(float(skews.max()), 3)

        # Categorical cardinality summary (informs target_encoding decisions)
        if cat_cols:
            cards = df_features[cat_cols].nunique()
            profile["categorical_max_cardinality"] = int(cards.max())
            profile["categorical_mean_cardinality"] = round(float(cards.mean()), 1)
            profile["high_card_categorical_count"] = int((cards >= 5).sum())

        return profile
    except Exception:
        # Profile is optional; never fail proposer because of it
        return {}


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
            "interactions": False,           # searchable
            "log_transform_spending": False, # searchable
            "target_encoding": False,        # searchable
            "group_aggregates": False,       # searchable (Kaggle top-20 signal)
        },
    },
    # CV config (lives in eval/cv.yaml on disk; mirrored here for LLM context).
    "cv": {
        "enabled": False,
        "n_splits": 5,
        "strategy": "stratified_kfold",
    },
    # Ensemble config (lives in train/ensemble.yaml on disk; mirrored here).
    "ensemble": {
        "enabled": False,
        "strategy": "voting_soft",
        "members": [],
        "stacking_meta_learner": "logistic_regression",
    },
}


def _load_node_secondary_metrics(node: Any) -> dict:
    """Best-effort load of secondary metrics (CV etc.) from node's metrics.json.

    Returns empty dict if file not found. Doesn't raise — used for prompt
    enrichment only, so missing metrics should degrade gracefully.
    """
    checkpoint = getattr(node, "checkpoint", None)
    if checkpoint is None:
        return {}
    ckpt_path = getattr(checkpoint, "path", None) if not isinstance(checkpoint, dict) else checkpoint.get("path")
    if not ckpt_path:
        return {}
    # checkpoint.path is .../nodes/<id>/workspace/checkpoints/models/<id>
    # metrics.json lives at    .../nodes/<id>/workspace/evolution/eval/full_state/test/metrics.json
    from pathlib import Path as _Path
    try:
        ws_path = _Path(ckpt_path).parents[2]  # .../workspace
        metrics_path = ws_path / "evolution" / "eval" / "full_state" / "test" / "metrics.json"
        if not metrics_path.exists():
            return {}
        with open(metrics_path) as f:
            data = json.load(f)
        # Return only the secondary signals we care about (keep LLM context lean)
        out = {}
        for k in ("cv_mean_accuracy", "cv_std", "cv_n_splits",
                  "ensemble_strategy", "ensemble_n_members", "ensemble_member_types"):
            if k in data:
                out[k] = data[k]
        return out
    except Exception:
        return {}


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

    # Apply patches in order. model/config.yaml, eval/cv.yaml, and
    # train/ensemble.yaml are all tracked and mirrored into the flat config.
    for n in chain:
        if not hasattr(n, "workspace_patch") or not n.workspace_patch:
            continue
        for op in n.workspace_patch.operations:
            if not op.key_path:
                continue
            if op.path == "model/config.yaml":
                _apply_patch_to_dict(config, op.key_path, op.value)
            elif op.path == "eval/cv.yaml":
                _apply_patch_to_dict(config.setdefault("cv", {}), op.key_path, op.value)
            elif op.path == "train/ensemble.yaml":
                _apply_patch_to_dict(config.setdefault("ensemble", {}), op.key_path, op.value)

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
        # Feature engineering flags — unexplored territory with high potential!
        "feature_engineering.flags.interactions": [True, False],
        "feature_engineering.flags.log_transform_spending": [True, False],
        "feature_engineering.flags.target_encoding": [True, False],
        "feature_engineering.flags.group_aggregates": [True, False],  # NEW
        # Cross-validation (eval/cv.yaml) — CV metric appears in secondary.
        "cv.enabled": [True, False],
        "cv.n_splits": [3, 5, 10],
        "cv.strategy": ["stratified_kfold", "kfold"],
        # Ensemble (train/ensemble.yaml) — combine multiple models for robustness.
        # When enabled, you MUST populate ensemble.members with at least 2 entries.
        "ensemble.enabled": [True, False],
        "ensemble.strategy": ["voting_soft", "voting_hard", "stacking"],
        "ensemble.stacking_meta_learner": ["logistic_regression", "ridge"],
    }

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region: str = "us-west-2",
        verbose: bool = True,
        workspace_root: str | Path | None = None,
    ):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.verbose = verbose
        # Path to seed workspace for loading train.csv to compute dataset profile.
        # Profile is objective stats (no competition name) injected into prompt.
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self._cached_profile: dict | None = None

    def _get_dataset_profile(self, parent_config: dict) -> dict:
        """Lazy-load and cache dataset profile from workspace."""
        if self._cached_profile is not None:
            return self._cached_profile
        if self._workspace_root is None:
            self._cached_profile = {}
            return self._cached_profile
        self._cached_profile = _compute_dataset_profile(
            workspace_root=self._workspace_root,
            train_filename=parent_config.get("train_data", "train.csv"),
            target_column=parent_config.get("target_column"),
            id_column=parent_config.get("id_column"),
        )
        return self._cached_profile

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
        """Apply a list of operations to a base config (for dedup check).

        Both model/config.yaml and eval/cv.yaml are mirrored into the same
        fingerprint dict (cv.yaml keys live under new_config["cv"]).
        """
        import copy
        new_config = copy.deepcopy(base_config)
        for op in operations:
            key_path = op.get("key_path", [])
            if not key_path:
                continue
            path = op.get("path", "model/config.yaml")
            if path == "eval/cv.yaml":
                _apply_patch_to_dict(new_config.setdefault("cv", {}), key_path, op.get("value"))
            elif path == "train/ensemble.yaml":
                _apply_patch_to_dict(new_config.setdefault("ensemble", {}), key_path, op.get("value"))
            else:
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
            secondary = _load_node_secondary_metrics(n)

            entry = {
                "node_id": n.node_id,
                "metric": round(n.metric, 5),  # primary = Kaggle holdout score
                "full_config": full_cfg,
                "is_best": False,
            }
            if secondary:
                entry["secondary"] = secondary  # may contain cv_mean_accuracy, cv_std
            history.append(entry)

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

        # Derive competition/dataset info from parent_config — keeps prompt
        # competition-agnostic and avoids hard-coding spaceship-titanic.
        competition_id = parent_config.get("competition_id", "unknown")
        task_type = parent_config.get("task_type", "classification")
        fe_flags = (
            parent_config.get("feature_engineering", {}).get("flags", {}) or {}
        )
        # Separate generic (base) FE flags from domain-specific ones
        GENERIC_FLAGS = {
            "drop_ids", "fill_numeric_median", "fill_categorical_mode",
            "fill_boolean_false", "log_transform_skewed", "target_encoding",
            "standard_scale", "label_encode_categoricals",
        }
        available_generic = [k for k in fe_flags.keys() if k in GENERIC_FLAGS]
        available_domain = [k for k in fe_flags.keys() if k not in GENERIC_FLAGS]

        return {
            "parent_config": parent_config,
            "parent_metric": parent.metric if hasattr(parent, "metric") else None,
            "metric_history": history,
            "best_metric": history[0]["metric"] if history else None,
            "tried_configs": list(tried_fingerprints),
            "crashed_configs": crashed_configs,
            "num_cycles_completed": len(valid_nodes),
            "search_space": self.PARAM_SPACE,
            # task_type is a user-provided label (classification|regression), not
            # a dataset identifier; safe to pass.
            "task_type": task_type,
            "generic_fe_flags": available_generic,
            "domain_fe_flags": available_domain,
            # Objective data-derived profile (no competition name)
            "dataset_profile": self._get_dataset_profile(parent_config),
        }

    def _build_prompt(self, context: dict) -> str:
        """Build prompt with full context and search space."""

        # Show full configs with metrics (not just descriptions!)
        # Include secondary metrics (cv_mean_accuracy, cv_std) when available —
        # gives the LLM a robustness signal orthogonal to the primary Kaggle score.
        def _entry_for_prompt(h):
            out = {
                "metric": h["metric"],  # primary = Kaggle 870-row holdout
                "config": h["full_config"],
                "is_best": h["is_best"],
            }
            if "secondary" in h:
                out["secondary"] = h["secondary"]  # cv_mean_accuracy ± cv_std on full train
            return out

        history_str = json.dumps(
            [_entry_for_prompt(h) for h in context["metric_history"]],
            indent=2,
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

        # Build dataset-agnostic hints from context
        domain_flags_str = (
            "Domain-specific FE flags available: " + ", ".join(context["domain_fe_flags"])
            if context["domain_fe_flags"]
            else "No domain-specific FE flags for this competition — use only generic flags."
        )
        generic_flags_str = ", ".join(context["generic_fe_flags"]) if context["generic_fe_flags"] else "(none)"

        # Format dataset profile as JSON (empty dict renders cleanly if no profile)
        profile = context.get("dataset_profile") or {}
        profile_str = (
            json.dumps(profile, indent=2) if profile else "(not available)"
        )

        return f"""You are an expert AutoML system optimizing a tabular ML model.

## Task
Task type: {context['task_type']}. Higher metric is better. Current best: {context['best_metric']}.

## Dataset profile (objective statistics from the training data)
{profile_str}

Use these statistics to guide proposals. Examples:
- Many high-skew numeric features → enable `log_transform_skewed`.
- High-cardinality categoricals → enable `target_encoding`.
- No missing values → fill strategy flags have no effect.
- Large imbalance in target_class_balance → consider sampling/weight tricks.
- Very few features but many rows → deeper trees may overfit less.

## 📏 Metric interpretation

The `metric` field is the **primary**: the official competition metric computed on
the mlebench test holdout. This is what MCGS optimizes.

When `secondary.cv_mean_accuracy` / `secondary.cv_std` appear in history, they are
**K-fold out-of-fold metric on the full training set** (from `eval/cv.yaml`). They
live on a different scale than the primary (typically lower, because the holdout
subsample can be easier/harder than the full train distribution). Use them as a
robustness signal, not as a replacement:

- **Primary↑ AND CV↑** → real improvement, trust it.
- **Primary↑ BUT CV↓ or CV-std↑** → likely overfitting the holdout; be skeptical.
- **Primary flat BUT CV↑** → genuine generalization gain; worth keeping.
- **High CV-std** → the config is unstable across folds — prefer tighter CV-std.

**Do NOT** interpret `cv.enabled=True` configs as "worse" just because their primary
metric is unchanged — CV's value is the `secondary` signal, not a primary-metric shift.

## ⚠️ What NOT to do
Recent losers:
{failures_summary if failures_summary else "  (none yet)"}

Avoid proposals that repeat a nearby region of a recent loser.

## 🎯 UNEXPLORED REGIONS WITH HIGH POTENTIAL
1. **Ensembles** (`ensemble.enabled=True`, path `train/ensemble.yaml`):
   - Typically adds +0.005 to +0.015 vs single model. Populate `ensemble.members`
     with 3-5 entries (each has `model_type`, `random_state`, `hyperparameters`).
     Diversity across model_type and/or random_state is what makes voting work.
   - See Example 3 below for the exact members list shape.
   - Strategies: voting_soft (stable), stacking (meta-learner on OOF).

2. **Cross-validation** (`cv.enabled=True`, path `eval/cv.yaml`):
   - Robustness signal via secondary metrics — does NOT improve primary directly.
   - Use when you want to confirm a jump is real vs. holdout noise.

3. **Feature engineering flags** (path `model/config.yaml`, key `feature_engineering.flags`):
   - Generic flags (available for ANY competition): {generic_flags_str}
     * `log_transform_skewed` — add log1p columns for high-skew numeric features.
     * `target_encoding` — smoothed mean-target encoding for high-cardinality categoricals.
     * `standard_scale` — z-score numeric columns (useful for linear models).
   - {domain_flags_str}
   - **IMPORTANT**: only toggle flags that exist in `parent_config.feature_engineering.flags`.
     Proposing a flag not in the parent config is a no-op.

4. **Model diversity + multi-parameter combos**: ensemble of lightgbm+xgboost with
   different seeds is often the single biggest win once single-model has plateaued.

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
    - "path": one of:
        * "model/config.yaml"       — for model_type, hyperparameters, feature_engineering
        * "eval/cv.yaml"            — for cv.enabled, cv.n_splits, cv.strategy
        * "train/ensemble.yaml"     — for ensemble.enabled, ensemble.strategy, ensemble.members, ensemble.stacking_meta_learner
    - "key_path": list of keys into that YAML (e.g. ["hyperparameters", "n_estimators"])
    - "value": the new value

Example 1 (Enable a generic FE flag — low-risk, available for any competition):
{{
  "reasoning": "Parent LightGBM is plateaued. log_transform_skewed=True applies log1p to high-skew numeric columns, often helps tree splits pick smoother thresholds. It's a generic flag available for any tabular task.",
  "base_node_id": "node-5a44e50433",
  "description": "LightGBM + log_transform_skewed=True (generic FE)",
  "operations": [
    {{"op": "replace", "path": "model/config.yaml", "key_path": ["feature_engineering", "flags", "log_transform_skewed"], "value": true}}
  ]
}}

Example 2 (Enable CV for robustness signal — primary metric unchanged):
{{
  "reasoning": "Best config at 0.832 holdout, but no CV signal yet. Enabling 5-fold CV exposes cv_mean_accuracy ± cv_std in history — a robustness check orthogonal to the 870-row holdout. Does NOT improve primary metric directly; provides signal for next cycle.",
  "base_node_id": "node-abc123",
  "description": "Enable 5-fold CV on current best (for robustness signal)",
  "operations": [
    {{"op": "replace", "path": "eval/cv.yaml", "key_path": ["enabled"], "value": true}},
    {{"op": "replace", "path": "eval/cv.yaml", "key_path": ["n_splits"], "value": 5}}
  ]
}}

Example 3 (Build ensemble — biggest single move above 0.82 baseline):
{{
  "reasoning": "Single models have plateaued around 0.832. Ensemble of 3 diverse models (2 LightGBM seeds + 1 XGBoost) with voting_soft typically adds +0.005 to +0.010 via variance reduction and model-type diversity.",
  "base_node_id": "node-a84d59df1c",
  "description": "voting_soft ensemble: 2x LightGBM (diff seeds) + 1x XGBoost",
  "operations": [
    {{"op": "replace", "path": "train/ensemble.yaml", "key_path": ["enabled"], "value": true}},
    {{"op": "replace", "path": "train/ensemble.yaml", "key_path": ["strategy"], "value": "voting_soft"}},
    {{"op": "replace", "path": "train/ensemble.yaml", "key_path": ["members"], "value": [
      {{"model_type": "lightgbm", "random_state": 42,  "hyperparameters": {{"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}}}},
      {{"model_type": "lightgbm", "random_state": 100, "hyperparameters": {{"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05}}}},
      {{"model_type": "xgboost",  "random_state": 42,  "hyperparameters": {{"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}}}}
    ]}}
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
