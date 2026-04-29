"""Feature engineering for tabular ML tasks.

Architecture:
  BaseTabularFeatureEngineer
    - Task-agnostic: works on any tabular dataset via dtypes-only logic.
    - Handles ID drop, missing values, categorical encoding, scaling,
      optional log-transform and target-encoding on auto-detected columns.
    - Configured via flags dict (same schema as before).

  SpaceshipTitanicFeatureEngineer(BaseTabularFeatureEngineer)
    - Inherits all base behavior.
    - Adds domain-specific features (Cabin split, spending, group aggregates,
      CryoSleep interactions).
    - Its flags extend the base flags with domain-specific ones.

  create_feature_engineer(competition_id, flags):
    Returns the specific class for known competitions; otherwise the base.
    New competitions can run out of the box with just the base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ──────────────────────────────────────────────────────────────────────────
# Base: task-agnostic tabular FE
# ──────────────────────────────────────────────────────────────────────────


class BaseTabularFeatureEngineer:
    """Generic tabular feature engineer for any classification/regression task.

    Flags (applied in order, each can be toggled):
      drop_ids                   — drop columns that look like IDs (high-cardinality unique)
      fill_numeric_median        — fill numeric NaN with per-column median
      fill_categorical_mode      — fill categorical NaN with mode or 'Unknown'
      fill_boolean_false         — fill boolean NaN with False
      log_transform_skewed       — add log1p columns for high-skew numeric (>1.0)
      target_encoding            — smoothed mean-target encode high-cardinality categoricals
      standard_scale             — apply StandardScaler to all numeric columns
      label_encode_categoricals  — LabelEncoder for remaining categoricals
    """

    DEFAULT_FLAGS = {
        "drop_ids": True,
        "fill_numeric_median": True,
        "fill_categorical_mode": True,
        "fill_boolean_false": True,
        "log_transform_skewed": False,    # searchable
        "target_encoding": False,         # searchable
        "standard_scale": False,          # searchable
        "label_encode_categoricals": True,
    }

    # Columns considered IDs (heuristic: high unique ratio, not target)
    ID_UNIQUE_RATIO_THRESHOLD = 0.95
    TARGET_ENCODE_HIGH_CARD_MIN = 5     # categoricals with at least this many unique levels
    SKEW_THRESHOLD = 1.0                # log-transform columns with |skew| > this
    SMOOTHING_WEIGHT = 10               # for target encoding

    def __init__(self, flags: dict | None = None):
        self.flags = {**self.DEFAULT_FLAGS, **(flags or {})}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.target_encoders: dict[str, dict] = {}
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        # Remember which columns were identified for log/target-encode at fit time
        self._log_transform_cols: list[str] = []
        self._target_encode_cols: list[str] = []
        self._id_cols: list[str] = []

    # ── Main pipeline ────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        df = df.copy()

        # Subclasses insert domain-specific features BEFORE generic processing.
        df = self._apply_domain_features(df, is_train=is_train, y=y)

        if self.flags.get("drop_ids", True):
            df = self._drop_id_columns(df, is_train=is_train)

        if self.flags.get("log_transform_skewed", False):
            df = self._log_transform_skewed(df, is_train=is_train)

        if self.flags.get("target_encoding", False):
            if is_train and y is not None:
                df = self._target_encode(df, y, is_train=True)
            elif not is_train and self.target_encoders:
                df = self._target_encode(df, y=None, is_train=False)

        if self.flags.get("fill_numeric_median", True):
            df = self._fill_numeric(df, is_train=is_train)

        if self.flags.get("fill_categorical_mode", True):
            df = self._fill_categorical(df)

        if self.flags.get("fill_boolean_false", True):
            df = self._fill_boolean(df)

        if self.flags.get("label_encode_categoricals", True):
            df = self._label_encode(df, is_train=is_train)

        if self.flags.get("standard_scale", False):
            df = self._standard_scale(df, is_train=is_train)

        if is_train:
            self.feature_names = df.columns.tolist()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df, is_train=False, y=None)

    # ── Extension hook for subclasses ────────────────────────────────

    def _apply_domain_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
        y: pd.Series | None,
    ) -> pd.DataFrame:
        """Override in subclasses to add domain-specific features BEFORE generic steps."""
        return df

    # ── Task-agnostic implementations ────────────────────────────────

    def _drop_id_columns(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Drop ID-like columns: non-numeric (string/object) with very high unique ratio."""
        if is_train:
            id_cols = []
            n = len(df)
            for col in df.columns:
                try:
                    unique_ratio = df[col].nunique(dropna=False) / max(n, 1)
                except Exception:
                    continue
                # "Non-numeric" captures both object dtype and newer pd.StringDtype.
                is_non_numeric = not pd.api.types.is_numeric_dtype(df[col])
                if unique_ratio >= self.ID_UNIQUE_RATIO_THRESHOLD and is_non_numeric:
                    id_cols.append(col)
            self._id_cols = id_cols
        to_drop = [c for c in self._id_cols if c in df.columns]
        return df.drop(columns=to_drop)

    def _log_transform_skewed(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Add log1p(col) for high-skew non-negative numeric columns."""
        if is_train:
            self._log_transform_cols = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_min = df[col].min()
                if pd.isna(col_min) or col_min < 0:
                    continue
                # Skew is defined, non-trivial?
                try:
                    skew = float(df[col].skew())
                except Exception:
                    continue
                if abs(skew) > self.SKEW_THRESHOLD:
                    self._log_transform_cols.append(col)
        for col in self._log_transform_cols:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].fillna(0))
        return df

    def _target_encode(
        self,
        df: pd.DataFrame,
        y: pd.Series | None,
        is_train: bool,
    ) -> pd.DataFrame:
        """Smoothed mean-target encoding on high-cardinality categoricals."""
        if is_train and y is not None:
            # Identify target-encode columns
            self._target_encode_cols = []
            for col in df.select_dtypes(include=["object", "category", "string"]).columns:
                n_unique = df[col].nunique(dropna=False)
                if n_unique >= self.TARGET_ENCODE_HIGH_CARD_MIN:
                    self._target_encode_cols.append(col)

            global_mean = float(y.mean())
            for col in self._target_encode_cols:
                if col not in df.columns:
                    continue
                grouped = y.groupby(df[col].fillna("Unknown")).agg(["mean", "count"])
                grouped["smoothed"] = (
                    grouped["mean"] * grouped["count"]
                    + global_mean * self.SMOOTHING_WEIGHT
                ) / (grouped["count"] + self.SMOOTHING_WEIGHT)
                self.target_encoders[col] = grouped["smoothed"].to_dict()
                self.target_encoders[f"{col}_global"] = global_mean

        # Apply
        for col in self._target_encode_cols:
            if col in df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                global_mean = self.target_encoders.get(f"{col}_global", 0.5)
                df[f"{col}_target_enc"] = (
                    df[col].fillna("Unknown").map(encoder).fillna(global_mean)
                )
        return df

    def _fill_numeric(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if is_train:
            self._numeric_medians = {c: float(df[c].median()) for c in numeric_cols}
        for col in numeric_cols:
            if df[col].isnull().any():
                med = self._numeric_medians.get(col, 0.0) if hasattr(self, "_numeric_medians") else float(df[col].median())
                df[col] = df[col].fillna(med)
        return df

    def _fill_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object", "category", "string"]).columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill = mode_val[0] if len(mode_val) > 0 else "Unknown"
                df[col] = df[col].fillna(fill)
        return df

    def _fill_boolean(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].fillna(False)
        return df

    def _label_encode(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        for col in list(df.select_dtypes(include=["object", "category", "string"]).columns):
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    # Never seen at train time; drop to avoid crashes
                    df = df.drop(columns=[col])
                    continue
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def _standard_scale(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        if is_train:
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            if self.scaler is not None:
                # Align columns at inference time
                existing = [c for c in numeric_cols if c in df.columns]
                df[existing] = self.scaler.transform(df[existing])
        return df


# ──────────────────────────────────────────────────────────────────────────
# Spaceship-Titanic specific
# ──────────────────────────────────────────────────────────────────────────


class SpaceshipTitanicFeatureEngineer(BaseTabularFeatureEngineer):
    """Spaceship-Titanic specific features layered on top of base engineer."""

    # Override with spaceship-specific flags PLUS inherit base flags
    DEFAULT_FLAGS = {
        **BaseTabularFeatureEngineer.DEFAULT_FLAGS,
        # Domain-specific features (all default True for back-compat)
        "passenger_id": True,
        "cabin_split": True,
        "spending_features": True,
        "age_groups": True,
        "family_features": True,
        # Domain-specific searchable flags
        "interactions": False,
        "log_transform_spending": False,   # domain-specific (explicit spending cols)
        "group_aggregates": False,
        # Override default: target_encoding is domain-aware for this dataset
        "target_encoding": False,
    }

    def __init__(self, flags: dict | None = None):
        super().__init__(flags=flags)
        self.group_sizes: dict = {}
        self.surname_counts: dict = {}

    # Hook from base: insert domain features first
    def _apply_domain_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
        y: pd.Series | None,
    ) -> pd.DataFrame:
        if self.flags.get("passenger_id", True):
            df = self._extract_passenger_id_features(df)
        elif "PassengerId" in df.columns:
            df = df.drop(columns=["PassengerId"])

        if self.flags.get("cabin_split", True):
            df = self._extract_cabin_features(df)
        elif "Cabin" in df.columns:
            df = df.drop(columns=["Cabin"])

        if self.flags.get("spending_features", True):
            df = self._create_spending_features(df)

        if self.flags.get("log_transform_spending", False):
            df = self._log_transform_spending(df)

        if self.flags.get("age_groups", True):
            df = self._create_age_groups(df)

        if self.flags.get("family_features", True):
            df = self._create_family_features(df, is_train=is_train)
        elif "Name" in df.columns:
            df = df.drop(columns=["Name"])

        if self.flags.get("interactions", False):
            df = self._create_interaction_features(df)

        if self.flags.get("group_aggregates", False):
            df = self._create_group_aggregates(df, is_train=is_train)

        # Drop the raw integer Group ID before it pollutes the model.
        # Used only as a join key in _create_group_aggregates.
        if "Group" in df.columns:
            df = df.drop(columns=["Group"])

        return df

    # ── Domain-specific implementations (unchanged from original) ────

    def _extract_passenger_id_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "PassengerId" in df.columns:
            split = df["PassengerId"].str.split("_", expand=True)
            df["Group"] = split[0].astype(int)
            df["GroupPosition"] = split[1].astype(int)
            df = df.drop(columns=["PassengerId"])
        return df

    def _extract_cabin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Cabin" in df.columns:
            split = df["Cabin"].str.split("/", expand=True)
            df["CabinDeck"] = split[0]
            df["CabinNum"] = pd.to_numeric(split[1], errors="coerce")
            df["CabinSide"] = split[2]
            df = df.drop(columns=["Cabin"])
        return df

    def _create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        if all(c in df.columns for c in cols):
            df["TotalSpent"] = df[cols].sum(axis=1)
            df["HasSpending"] = (df["TotalSpent"] > 0).astype(int)
            df["NumServicesUsed"] = (df[cols] > 0).sum(axis=1)
            arr = df[cols].values
            arr_nz = np.where(arr > 0, arr, np.nan)
            df["AvgSpendingPerService"] = np.nanmean(arr_nz, axis=1)
            df["AvgSpendingPerService"] = df["AvgSpendingPerService"].fillna(0)
        return df

    def _log_transform_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpent"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].fillna(0))
        return df

    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Age" in df.columns:
            df["AgeGroup"] = pd.cut(
                df["Age"],
                bins=[0, 12, 18, 25, 35, 50, 65, 100],
                labels=["Child", "Teen", "YoungAdult", "Adult", "MiddleAged", "Senior", "Elderly"],
            )
            df["AgeGroup"] = df["AgeGroup"].astype(str).replace("nan", "Unknown")
        return df

    def _create_family_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        if "Group" in df.columns:
            if is_train:
                self.group_sizes = df.groupby("Group").size().to_dict()
            df["FamilySize"] = df["Group"].map(self.group_sizes)
            df["TravelingAlone"] = (df["FamilySize"] == 1).astype(int)
            df["LargeFamily"] = (df["FamilySize"] >= 4).astype(int)

        if "Name" in df.columns:
            df["Surname"] = df["Name"].str.split().str[-1]
            if is_train:
                self.surname_counts = df["Surname"].value_counts().to_dict()
            df["SurnameCount"] = df["Surname"].map(self.surname_counts).fillna(1)
            df = df.drop(columns=["Name", "Surname"])
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "CryoSleep" in df.columns and "HasSpending" in df.columns:
            cryo = df["CryoSleep"].astype(str).map({"True": 1, "False": 0, "nan": 0}).fillna(0)
            df["Cryo_x_Spending"] = cryo * df["HasSpending"]
        if "Age" in df.columns and "TotalSpent" in df.columns:
            df["Age_x_Spent"] = df["Age"].fillna(df["Age"].median()) * df["TotalSpent"].fillna(0)
        if "VIP" in df.columns and "TotalSpent" in df.columns:
            vip = df["VIP"].astype(str).map({"True": 1, "False": 0, "nan": 0}).fillna(0)
            df["VIP_x_Spent"] = vip * df["TotalSpent"].fillna(0)
        if "FamilySize" in df.columns and "TotalSpent" in df.columns:
            df["Family_x_Spent"] = df["FamilySize"] * df["TotalSpent"].fillna(0)
        return df

    def _create_group_aggregates(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        if "Group" not in df.columns:
            return df
        if "TotalSpent" in df.columns:
            df["GroupSpendMean"] = df.groupby("Group")["TotalSpent"].transform("mean")
            df["GroupSpendStd"] = df.groupby("Group")["TotalSpent"].transform("std").fillna(0)
            df["GroupSpendMax"] = df.groupby("Group")["TotalSpent"].transform("max")
        if "CryoSleep" in df.columns:
            cryo = df["CryoSleep"].astype(str).map(
                {"True": 1.0, "False": 0.0, "nan": 0.5, "NaN": 0.5}
            ).fillna(0.5).astype(float)
            df["GroupCryoRate"] = cryo.groupby(df["Group"]).transform("mean")
        if "Age" in df.columns:
            df["GroupAgeMean"] = df.groupby("Group")["Age"].transform("mean")
        if "FamilySize" not in df.columns:
            df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
        return df


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────


def create_feature_engineer(
    competition_id: str,
    flags: dict | None = None,
) -> BaseTabularFeatureEngineer:
    """Factory: return a domain-aware engineer if we know the competition,
    otherwise the generic tabular engineer."""
    if competition_id == "spaceship-titanic":
        return SpaceshipTitanicFeatureEngineer(flags=flags)
    # Generic fallback for any other tabular competition.
    return BaseTabularFeatureEngineer(flags=flags)


__all__ = [
    "BaseTabularFeatureEngineer",
    "SpaceshipTitanicFeatureEngineer",
    "create_feature_engineer",
]
