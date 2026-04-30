"""Sklearn/XGBoost Backend for Traditional ML Training

This backend trains traditional ML models (RandomForest, XGBoost, LightGBM)
instead of LLMs, enabling TrainingEvolver to work as an AutoML framework.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CVEnsembleModel:
    """Wraps K fold-trained models. Predictions are the mean across folds."""

    def __init__(self, models: list, cv_mean_score: float, cv_std: float, n_splits: int):
        self.models = models
        self.cv_mean_score = cv_mean_score
        self.cv_std = cv_std
        self.n_splits = n_splits

    def predict_proba(self, X):
        # Mean of per-fold predicted probabilities
        probas = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probas

    def predict(self, X):
        if hasattr(self.models[0], "predict_proba"):
            probas = self.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] == 2:
                return (probas[:, 1] > 0.5).astype(int)
            return np.argmax(probas, axis=1)
        # Regression fallback: mean of predictions
        return np.mean([m.predict(X) for m in self.models], axis=0)


class EnsembleModel:
    """Combines multiple trained base models via voting or stacking.

    - voting_soft: average predict_proba across members
    - voting_hard: majority vote on predicted labels
    - stacking: meta-learner trained on K-fold OOF predictions of members
    """

    def __init__(
        self,
        members: list,
        strategy: str = "voting_soft",
        meta_learner=None,
        member_specs: list | None = None,
    ):
        self.members = members
        self.strategy = strategy
        self.meta_learner = meta_learner  # fitted only when strategy=stacking
        self.member_specs = member_specs or []  # for introspection

    @property
    def _is_regression(self) -> bool:
        # Members without predict_proba are regressors.
        return not hasattr(self.members[0], "predict_proba")

    def predict_proba(self, X):
        if self._is_regression:
            # Regression: no probabilities. Caller should use .predict() instead.
            raise AttributeError(
                "EnsembleModel: regression members do not support predict_proba"
            )
        if self.strategy in ("voting_soft", "voting_hard"):
            probas = np.mean([m.predict_proba(X) for m in self.members], axis=0)
            return probas
        elif self.strategy == "stacking":
            meta_features = self._stack_features(X)
            if hasattr(self.meta_learner, "predict_proba"):
                return self.meta_learner.predict_proba(meta_features)
            preds = self.meta_learner.predict(meta_features)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds.astype(int)] = 1.0
            return proba
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")

    def predict(self, X):
        if self._is_regression:
            # Mean of regressor outputs (voting_soft / voting_hard collapse to the same op)
            return np.mean([m.predict(X) for m in self.members], axis=0)
        if self.strategy == "voting_hard":
            member_preds = np.array([m.predict(X) for m in self.members])
            from scipy.stats import mode
            return mode(member_preds, axis=0, keepdims=False).mode
        # voting_soft and stacking use predict_proba threshold (classification)
        probas = self.predict_proba(X)
        if probas.ndim == 2 and probas.shape[1] == 2:
            return (probas[:, 1] > 0.5).astype(int)
        return np.argmax(probas, axis=1)

    def _stack_features(self, X):
        """Build meta-features for stacking."""
        cols = []
        for m in self.members:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
                # Binary: proba of class 1; multi-class: full matrix
                if p.ndim == 2 and p.shape[1] == 2:
                    cols.append(p[:, 1:2])
                else:
                    cols.append(p)
            else:
                # Regressor meta-feature: its prediction as a single column
                preds = m.predict(X)
                cols.append(preds.reshape(-1, 1))
        return np.hstack(cols)


class SklearnBackend:
    """Backend for training traditional ML models on tabular data."""

    def __init__(self, mock: bool = False):
        self.mock = mock

    def run_trial(
        self,
        workspace: Any,
        node: Any,
        budget: Dict[str, Any],
        benchmark: Any,
    ) -> Any:
        """Run a complete trial: train ML model + evaluate.

        Args:
            workspace: Training workspace with ML config
            node: MCGS graph node
            budget: Resource budget (time, etc.)
            benchmark: MLE-Bench benchmark adapter

        Returns:
            TrainingTrialResult with metrics
        """
        from ..training.types import TrainingTrialResult, CheckpointRef, TrialStatus

        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)

        try:
            # 1. Load configuration
            config = self._load_ml_config(workspace)
            cv_config = self._load_cv_config(workspace)
            ensemble_config = self._load_ensemble_config(workspace)

            # Explicit task_type from config takes precedence over heuristics
            self._task_type = config.get("task_type")  # "classification" | "regression" | None

            # 2. Load and prepare data
            X_train, y_train, X_test = self._load_data(workspace, config)

            # 3. Train: ensemble > CV > single (in priority order)
            self._cv_mean_score = None
            self._cv_std = None
            self._ensemble_info = None

            if ensemble_config.get("enabled", False) and ensemble_config.get("members"):
                model = self._train_ensemble(config, ensemble_config, X_train, y_train)
                self._ensemble_info = {
                    "strategy": ensemble_config.get("strategy", "voting_soft"),
                    "n_members": len(model.members),
                    "member_types": [s.get("model_type", "?") for s in model.member_specs],
                }
                print(f"✓ Ensemble: {self._ensemble_info['strategy']} over "
                      f"{self._ensemble_info['n_members']} members "
                      f"({self._ensemble_info['member_types']})")
            elif cv_config.get("enabled", False):
                model = self._train_model_with_cv(config, cv_config, X_train, y_train)
                self._cv_mean_score = model.cv_mean_score
                self._cv_std = model.cv_std
                print(f"✓ CV-mean accuracy: {self._cv_mean_score:.5f} ± {self._cv_std:.5f} "
                      f"(n_splits={model.n_splits})")
            else:
                model = self._train_model(config, X_train, y_train)

            # 4. Save model (checkpoint)
            checkpoint_path = self._save_model(workspace, node.node_id, model, config, self.feature_engineer)
            checkpoint = CheckpointRef(
                name=f"model-{node.node_id}",
                path=str(checkpoint_path),
                kind="full_state",
                metadata={"model_type": config.get("model_type", "unknown")},
            )

            # 5. Evaluate on test set
            result_dir = benchmark.evaluate(
                workspace=workspace,
                checkpoint=checkpoint,
                backend=self,
                split="test",
            )

            # 6. Parse metrics
            eval_metrics = benchmark.parse_metrics(result_dir)

            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=str(workspace_path),
                status="success",
                checkpoint=checkpoint,
                eval_metrics=eval_metrics,
                error_buckets=benchmark.analyze_errors(result_dir, eval_metrics),
                cost={"training_time": 0.0},  # TODO: track actual time
            )

        except Exception as e:
            workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=str(workspace_path),
                status="train_failed",
                train_metrics={"error": str(e)},
            )

    def _load_ml_config(self, workspace: Any) -> Dict[str, Any]:
        """Load ML model configuration from workspace."""
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        config_path = workspace_path / "model" / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    def _load_cv_config(self, workspace: Any) -> Dict[str, Any]:
        """Load CV configuration from eval/cv.yaml. Returns disabled config if file missing."""
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        cv_path = workspace_path / "eval" / "cv.yaml"

        if not cv_path.exists():
            return {"enabled": False}

        with open(cv_path) as f:
            return yaml.safe_load(f) or {"enabled": False}

    def _load_ensemble_config(self, workspace: Any) -> Dict[str, Any]:
        """Load ensemble configuration from train/ensemble.yaml."""
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        ens_path = workspace_path / "train" / "ensemble.yaml"

        if not ens_path.exists():
            return {"enabled": False, "members": []}

        with open(ens_path) as f:
            return yaml.safe_load(f) or {"enabled": False, "members": []}

    def _load_data(self, workspace: Any, config: Dict) -> tuple:
        """Load training and test data."""
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        data_dir = workspace_path / "data"

        # Load from MLE-Bench prepared data
        # Assume data paths are in config
        train_path = config.get("train_data", "train.csv")
        test_path = config.get("test_data", "test.csv")
        target_col = config.get("target_column", "target")

        # Resolve paths
        if not Path(train_path).is_absolute():
            train_path = data_dir / train_path
        if not Path(test_path).is_absolute():
            test_path = data_dir / test_path

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Get ID column (usually 'PassengerId', 'id', etc.)
        id_col = config.get("id_column", "PassengerId")

        # Store test IDs for later submission
        self.test_ids = test_df[id_col].values if id_col in test_df.columns else test_df.index.values

        # Separate target
        y_train = train_df[target_col]

        # Remove target only; KEEP PassengerId so FE can extract Group features.
        # FE's _extract_passenger_id_features drops PassengerId after extracting Group.
        X_train_raw = train_df.drop(columns=[target_col])
        X_test_raw = test_df.copy()

        # Save y for target_encoding in FE
        self._current_y = y_train

        # Feature engineering (advanced or basic)
        X_train, X_test, feature_engineer = self._apply_feature_engineering(X_train_raw, X_test_raw, config)

        # Store feature engineer for later use in evaluation
        self.feature_engineer = feature_engineer

        return X_train, y_train, X_test

    def _apply_feature_engineering(self, X_train, X_test, config):
        """Apply feature engineering steps.

        Returns:
            X_train, X_test, feature_engineer (or None if basic FE)
        """
        fe_config = config.get("feature_engineering", {})
        competition_id = config.get("competition_id", "unknown")

        # Check if we should use advanced feature engineering
        use_advanced_fe = fe_config.get("advanced", True)

        if use_advanced_fe:
            # Try to use competition-specific feature engineering
            try:
                from .feature_engineering import create_feature_engineer

                # Read feature flags from config (for LLM-driven FE mutation)
                fe_flags = fe_config.get("flags", None)

                fe = create_feature_engineer(competition_id, flags=fe_flags)
                if fe is not None:
                    enabled = (
                        [k for k, v in fe.flags.items() if v]
                        if hasattr(fe, "flags") else []
                    )
                    print(f"✓ Using advanced FE for {competition_id} | enabled: {enabled}")
                    # Pass y to support target_encoding
                    X_train = fe.fit_transform(X_train, is_train=True, y=self._current_y)
                    X_test = fe.transform(X_test)
                    return X_train, X_test, fe
            except Exception as e:
                print(f"Warning: Advanced FE failed ({e}), falling back to basic FE")

        # Fallback to basic feature engineering
        print(f"Using basic feature engineering")

        # Separate numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # Handle missing values for numeric columns
        fillna_strategy = fe_config.get("fillna", "mean")
        if numeric_cols:
            if fillna_strategy == "mean":
                X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
                X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())
            elif fillna_strategy == "median":
                X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
                X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
            elif fillna_strategy == "zero":
                X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
                X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

        # Handle missing values for categorical columns
        if categorical_cols:
            for col in categorical_cols:
                # Fill with mode or 'Unknown'
                mode_val = X_train[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)

        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            # Handle unseen labels in test set
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X_test[col] = le.transform(X_test[col].astype(str))

        # Scaling (only on numeric columns after encoding)
        if fe_config.get("scale", False):
            all_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if all_numeric_cols:
                scaler = StandardScaler()
                X_train[all_numeric_cols] = scaler.fit_transform(X_train[all_numeric_cols])
                X_test[all_numeric_cols] = scaler.transform(X_test[all_numeric_cols])

        return X_train, X_test, None

    def _train_model(self, config: Dict, X_train, y_train):
        """Train ML model based on config.

        GPU is auto-enabled for xgboost/lightgbm when:
          - config.get("use_gpu") is True (explicit), OR
          - training set has >= 50k rows (heuristic; GPU pays off at scale)
        GPU can be force-disabled with config["use_gpu"] = False.
        """
        model_type = config.get("model_type", "random_forest")
        hyperparams = dict(config.get("hyperparameters", {}))  # copy; we may inject GPU params

        # GPU auto-enable heuristic
        use_gpu_explicit = config.get("use_gpu", None)
        if use_gpu_explicit is True:
            gpu = True
        elif use_gpu_explicit is False:
            gpu = False
        else:
            gpu = len(X_train) >= 50_000

        if model_type == "random_forest":
            # No GPU support in sklearn RF
            if self._is_classification(y_train):
                model = RandomForestClassifier(**hyperparams)
            else:
                model = RandomForestRegressor(**hyperparams)

        elif model_type == "xgboost":
            import xgboost as xgb
            if gpu:
                # xgboost >= 2.0 uses device="cuda"; older uses tree_method="gpu_hist"
                hyperparams.setdefault("device", "cuda")
                hyperparams.setdefault("tree_method", "hist")
            if self._is_classification(y_train):
                model = xgb.XGBClassifier(**hyperparams)
            else:
                model = xgb.XGBRegressor(**hyperparams)

        elif model_type == "lightgbm":
            import lightgbm as lgb
            if gpu:
                hyperparams.setdefault("device_type", "gpu")
            if self._is_classification(y_train):
                model = lgb.LGBMClassifier(**hyperparams)
            else:
                model = lgb.LGBMRegressor(**hyperparams)

        elif model_type == "logistic_regression":
            model = LogisticRegression(**hyperparams)

        elif model_type == "ridge":
            model = Ridge(**hyperparams)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model.fit(X_train, y_train)

        return model

    def _train_model_with_cv(self, config: Dict, cv_config: Dict, X_train, y_train) -> CVEnsembleModel:
        """Train K fold-models via CV. Returns CVEnsembleModel whose predictions average the folds."""
        n_splits = int(cv_config.get("n_splits", 5))
        strategy = cv_config.get("strategy", "stratified_kfold")
        shuffle = bool(cv_config.get("shuffle", True))
        random_state = cv_config.get("random_state", 42)

        if strategy == "stratified_kfold" and self._is_classification(y_train):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        fold_models = []
        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train, y_train)):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            # Train one fold with the same hyperparams
            fold_model = self._train_model(config, X_tr, y_tr)

            # Out-of-fold score — this is the signal we report as primary metric
            if hasattr(fold_model, "predict_proba") and self._is_classification(y_tr):
                va_preds = (fold_model.predict_proba(X_va)[:, 1] > 0.5).astype(int)
                score = accuracy_score(y_va, va_preds)
            else:
                va_preds = fold_model.predict(X_va)
                score = accuracy_score(y_va, va_preds) if self._is_classification(y_tr) else -np.mean((va_preds - y_va) ** 2)

            fold_models.append(fold_model)
            fold_scores.append(score)

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))
        return CVEnsembleModel(
            models=fold_models,
            cv_mean_score=cv_mean,
            cv_std=cv_std,
            n_splits=n_splits,
        )

    def _train_ensemble(self, base_config: Dict, ensemble_config: Dict, X_train, y_train) -> EnsembleModel:
        """Train each member with its own sub-config; wrap in EnsembleModel.

        Each member spec is a dict with:
          - model_type: str
          - hyperparameters: dict  (merged over base_config hyperparameters)
          - random_state: int (optional; falls back to base_config random_state)
        """
        strategy = ensemble_config.get("strategy", "voting_soft")
        members_spec = ensemble_config.get("members", []) or []

        if not members_spec:
            raise ValueError("ensemble.enabled=True but ensemble.members is empty")

        base_hparams = base_config.get("hyperparameters", {})

        base_model_type = base_config.get("model_type")

        trained_members = []
        for i, spec in enumerate(members_spec):
            member_type = spec.get("model_type", base_model_type)
            # Only inherit base_hparams when the member uses the same model_type;
            # hyperparameters like learning_rate are specific to the base model
            # and would crash a different algorithm (e.g. RF doesn't take learning_rate).
            if member_type == base_model_type:
                member_hparams = {**base_hparams, **(spec.get("hyperparameters") or {})}
            else:
                member_hparams = dict(spec.get("hyperparameters") or {})
            if "random_state" in spec:
                member_hparams["random_state"] = spec["random_state"]

            member_cfg = {
                "model_type": member_type,
                "hyperparameters": member_hparams,
            }
            print(f"  [ensemble member {i+1}/{len(members_spec)}] "
                  f"{member_cfg['model_type']} "
                  f"seed={member_hparams.get('random_state', 'default')}")
            trained_members.append(self._train_model(member_cfg, X_train, y_train))

        # Stacking: need K-fold OOF predictions to train the meta-learner
        meta_learner = None
        if strategy == "stacking":
            meta_learner = self._train_stacking_meta(
                members_spec=members_spec,
                base_config=base_config,
                X_train=X_train,
                y_train=y_train,
                meta_learner_name=ensemble_config.get("stacking_meta_learner", "logistic_regression"),
            )

        return EnsembleModel(
            members=trained_members,
            strategy=strategy,
            meta_learner=meta_learner,
            member_specs=members_spec,
        )

    def _train_stacking_meta(self, members_spec, base_config, X_train, y_train, meta_learner_name):
        """Train meta-learner on K-fold OOF proba predictions from each member."""
        n_splits = 5
        splitter = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    if self._is_classification(y_train)
                    else KFold(n_splits=n_splits, shuffle=True, random_state=42))

        base_hparams = base_config.get("hyperparameters", {})
        base_model_type = base_config.get("model_type")
        # OOF matrix: (N, M) where M = number of members (proba of class 1 per member)
        n = len(X_train)
        m = len(members_spec)
        oof = np.zeros((n, m))

        for member_idx, spec in enumerate(members_spec):
            member_type = spec.get("model_type", base_model_type)
            if member_type == base_model_type:
                member_hparams = {**base_hparams, **(spec.get("hyperparameters") or {})}
            else:
                member_hparams = dict(spec.get("hyperparameters") or {})
            if "random_state" in spec:
                member_hparams["random_state"] = spec["random_state"]
            member_cfg = {
                "model_type": member_type,
                "hyperparameters": member_hparams,
            }
            for tr_idx, va_idx in splitter.split(X_train, y_train):
                fold_model = self._train_model(member_cfg, X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                if hasattr(fold_model, "predict_proba"):
                    oof[va_idx, member_idx] = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
                else:
                    oof[va_idx, member_idx] = fold_model.predict(X_train.iloc[va_idx])

        # Train meta-learner on OOF features
        if meta_learner_name == "ridge":
            meta = Ridge()
        else:
            meta = LogisticRegression(max_iter=1000)
        meta.fit(oof, y_train)
        return meta

    def _is_classification(self, y):
        """Detect if task is classification or regression.

        Precedence:
          1. self._task_type (set from config.yaml::task_type if present)
          2. Heuristic: low unique ratio or object dtype
        """
        task_type = getattr(self, "_task_type", None)
        if task_type == "classification":
            return True
        if task_type == "regression":
            return False
        # Fallback heuristic
        unique_ratio = len(np.unique(y)) / len(y)
        return unique_ratio < 0.05 or y.dtype == "object"

    def _save_model(self, workspace: Any, node_id: str, model, config: Dict, feature_engineer=None) -> Path:
        """Save trained model as checkpoint."""
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)
        checkpoint_dir = workspace_path / "checkpoints" / "models" / node_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save feature engineer if provided
        if feature_engineer is not None:
            fe_path = checkpoint_dir / "feature_engineer.pkl"
            with open(fe_path, "wb") as f:
                pickle.dump(feature_engineer, f)

        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return checkpoint_dir

    def run_eval_plan(
        self,
        workspace: Any,
        checkpoint: Any,
        plan: Any,
    ) -> Path:
        """Run evaluation: load model, predict, save submission.

        Args:
            workspace: Workspace
            checkpoint: Trained model checkpoint
            plan: Eval plan from benchmark

        Returns:
            Path to result directory with submission.csv and metrics.json
        """
        # Load model
        model_path = Path(checkpoint.path) / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load feature engineer if available
        fe_path = Path(checkpoint.path) / "feature_engineer.pkl"
        feature_engineer = None
        if fe_path.exists():
            with open(fe_path, "rb") as f:
                feature_engineer = pickle.load(f)

        # Load test data from plan metadata
        workspace_path = Path(workspace.root) if hasattr(workspace, "root") else Path(workspace)

        # Get data paths and columns from plan metadata
        test_data_path = plan.metadata.get("test_data_path")
        if not test_data_path:
            # Fallback to default
            test_data_path = workspace_path / "data" / "test.csv"

        X_test = pd.read_csv(test_data_path)

        # Get ID column
        id_col = plan.metadata.get("id_column", "PassengerId")
        test_ids = X_test[id_col] if id_col in X_test.columns else X_test.index

        # KEEP ID column so FE can extract Group features (matches _load_data).
        # FE itself drops PassengerId after extracting features.
        X_test_raw = X_test.copy()

        # Apply feature engineering
        if feature_engineer is not None:
            # Use saved feature engineer
            X_test_features = feature_engineer.transform(X_test_raw)
        else:
            # Fallback: basic feature engineering
            X_test_features = X_test_raw

            # Separate numeric and categorical columns
            numeric_cols = X_test_features.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_test_features.select_dtypes(include=["object", "category"]).columns.tolist()

            # Handle missing values for numeric columns
            if numeric_cols:
                X_test_features[numeric_cols] = X_test_features[numeric_cols].fillna(X_test_features[numeric_cols].median())

            # Handle missing values and encode categorical columns
            if categorical_cols:
                for col in categorical_cols:
                    # Fill missing with 'Unknown'
                    X_test_features[col] = X_test_features[col].fillna('Unknown')
                    # Encode
                    le = LabelEncoder()
                    X_test_features[col] = le.fit_transform(X_test_features[col].astype(str))

        # Predict
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_test_features)
        else:
            predictions = model.predict(X_test_features)

        # Create submission
        result_dir = Path(plan.output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        submission_path = result_dir / "submission.csv"

        # Format depends on task type
        target_col = plan.metadata.get("target_column", "target")

        if predictions.ndim == 2 and predictions.shape[1] == 2:
            # Binary classification with probabilities - convert to class labels
            # Use class 1 probability > 0.5 threshold
            pred_labels = (predictions[:, 1] > 0.5).astype(bool)
            submission_df = pd.DataFrame({
                id_col: test_ids,
                target_col: pred_labels,
            })
        elif predictions.ndim == 2 and predictions.shape[1] > 2:
            # Multi-class classification with probabilities
            submission_df = pd.DataFrame(predictions, columns=[f"class_{i}" for i in range(predictions.shape[1])])
            submission_df.insert(0, id_col, test_ids)
        else:
            # Regression or already discrete predictions
            submission_df = pd.DataFrame({
                id_col: test_ids,
                target_col: predictions,
            })

        submission_df.to_csv(submission_path, index=False)

        # If CV was used, write metrics.json with CV-mean as primary metric.
        # The benchmark's _grade_submission runs AFTER run_eval_plan returns,
        # so it will overwrite this. We need to write AFTER the grader runs,
        # which happens in benchmark.evaluate. Instead, we signal via a marker
        # file that the caller (benchmark.evaluate) checks.
        cv_mean = getattr(self, "_cv_mean_score", None)
        if cv_mean is not None:
            marker_path = result_dir / "cv_mean.json"
            with open(marker_path, "w") as f:
                json.dump({
                    "cv_mean_accuracy": cv_mean,
                    "cv_std": getattr(self, "_cv_std", 0.0),
                    "n_splits": getattr(model, "n_splits", None),
                }, f)

        ens_info = getattr(self, "_ensemble_info", None)
        if ens_info is not None:
            ens_marker = result_dir / "ensemble.json"
            with open(ens_marker, "w") as f:
                json.dump(ens_info, f)

        return result_dir


__all__ = ["SklearnBackend", "CVEnsembleModel"]
