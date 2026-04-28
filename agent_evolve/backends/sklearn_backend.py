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
from sklearn.preprocessing import StandardScaler, LabelEncoder


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

            # 2. Load and prepare data
            X_train, y_train, X_test = self._load_data(workspace, config)

            # 3. Train ML model
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

        # Remove target and ID from dataframes before feature engineering
        X_train_raw = train_df.drop(columns=[target_col] + ([id_col] if id_col in train_df.columns else []))
        X_test_raw = test_df.drop(columns=[id_col] if id_col in test_df.columns else [])

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
        """Train ML model based on config."""
        model_type = config.get("model_type", "random_forest")
        hyperparams = config.get("hyperparameters", {})

        if model_type == "random_forest":
            if self._is_classification(y_train):
                model = RandomForestClassifier(**hyperparams)
            else:
                model = RandomForestRegressor(**hyperparams)

        elif model_type == "xgboost":
            import xgboost as xgb
            if self._is_classification(y_train):
                model = xgb.XGBClassifier(**hyperparams)
            else:
                model = xgb.XGBRegressor(**hyperparams)

        elif model_type == "lightgbm":
            import lightgbm as lgb
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

    def _is_classification(self, y):
        """Detect if task is classification or regression."""
        # Simple heuristic
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

        # Drop ID column
        X_test_raw = X_test.drop(columns=[id_col] if id_col in X_test.columns else [])

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

        return result_dir


__all__ = ["SklearnBackend"]
