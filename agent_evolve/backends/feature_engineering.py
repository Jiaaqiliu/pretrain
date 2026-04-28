"""Advanced feature engineering for Spaceship Titanic.

This module provides sophisticated feature engineering that goes beyond
simple fillna + LabelEncoder to extract meaningful patterns from the data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class SpaceshipTitanicFeatureEngineer:
    """Feature engineering specifically for Spaceship Titanic competition.

    Each component can be toggled via flags dict — enabling LLM-driven
    feature engineering search.
    """

    DEFAULT_FLAGS = {
        "passenger_id": True,       # Group + GroupPosition
        "cabin_split": True,        # Deck/Num/Side
        "spending_features": True,  # TotalSpent, HasSpending, etc.
        "age_groups": True,         # Bin Age into groups
        "family_features": True,    # FamilySize, TravelingAlone, SurnameCount
        "interactions": False,      # Age*TotalSpent, CryoSleep*HasSpending
        "log_transform_spending": False,  # log1p on skewed spending cols
        "target_encoding": False,   # mean target encoding for HomePlanet/Destination/Deck
        "group_aggregates": False,  # Group-level aggregates: GroupSpendMean, GroupCryoRate, etc.
    }

    def __init__(self, flags: dict | None = None):
        self.label_encoders = {}
        self.feature_names = []
        self.flags = {**self.DEFAULT_FLAGS, **(flags or {})}
        self.target_encoders = {}

    def fit_transform(self, df: pd.DataFrame, is_train: bool = True,
                      y: pd.Series | None = None) -> pd.DataFrame:
        """Transform features for training set.

        Args:
            df: Input dataframe
            is_train: Whether this is training data
            y: Target values (needed for target_encoding)
        """
        df = df.copy()

        # 1. Extract PassengerId features (gated)
        if self.flags.get("passenger_id", True):
            df = self._extract_passenger_id_features(df)
        elif "PassengerId" in df.columns:
            df = df.drop("PassengerId", axis=1)

        # 2. Extract Cabin features (gated)
        if self.flags.get("cabin_split", True):
            df = self._extract_cabin_features(df)
        elif "Cabin" in df.columns:
            df = df.drop("Cabin", axis=1)

        # 3. Create spending features (gated)
        if self.flags.get("spending_features", True):
            df = self._create_spending_features(df)

        # 3b. NEW: Log transform on spending (gated)
        if self.flags.get("log_transform_spending", False):
            df = self._log_transform_spending(df)

        # 4. Create age groups (gated)
        if self.flags.get("age_groups", True):
            df = self._create_age_groups(df)

        # 5. Create family features (gated)
        if self.flags.get("family_features", True):
            df = self._create_family_features(df, is_train=is_train)
        elif "Name" in df.columns:
            df = df.drop("Name", axis=1)

        # 5b. NEW: Interaction features (gated)
        if self.flags.get("interactions", False):
            df = self._create_interaction_features(df)

        # 5d. NEW: Group-level aggregates (gated) — the Kaggle top-20 signal
        if self.flags.get("group_aggregates", False):
            df = self._create_group_aggregates(df, is_train=is_train)

        # IMPORTANT: The raw 'Group' integer ID is pure noise (sequential IDs with no
        # semantic meaning). It's used only as a join key for aggregates above.
        # Drop before training to avoid polluting models.
        if "Group" in df.columns:
            df = df.drop("Group", axis=1)

        # 5c. NEW: Target encoding (gated). On test, uses saved encoders from train.
        if self.flags.get("target_encoding", False):
            if is_train and y is not None:
                df = self._target_encode(df, y, is_train=True)
            elif not is_train and self.target_encoders:
                df = self._target_encode(df, y=None, is_train=False)

        # 6. Handle missing values intelligently
        df = self._handle_missing_values(df)

        # 7. Encode categorical features
        df = self._encode_categorical_features(df, is_train=is_train)

        if is_train:
            self.feature_names = df.columns.tolist()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features for test set (no target available)."""
        return self.fit_transform(df, is_train=False, y=None)

    def _extract_passenger_id_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract group and position from PassengerId.

        Format: gggg_pp where gggg is group ID, pp is position in group.
        """
        if 'PassengerId' in df.columns:
            # Split PassengerId into group and position
            passenger_split = df['PassengerId'].str.split('_', expand=True)
            df['Group'] = passenger_split[0].astype(int)
            df['GroupPosition'] = passenger_split[1].astype(int)

            # Drop original PassengerId (not a feature)
            df = df.drop('PassengerId', axis=1)

        return df

    def _extract_cabin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Deck, Num, Side from Cabin.

        Format: Deck/Num/Side (e.g., F/906/P)
        """
        if 'Cabin' in df.columns:
            # Split Cabin
            cabin_split = df['Cabin'].str.split('/', expand=True)
            df['CabinDeck'] = cabin_split[0]
            df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
            df['CabinSide'] = cabin_split[2]

            # Drop original Cabin
            df = df.drop('Cabin', axis=1)

        return df

    def _create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spending-related features."""
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        if all(col in df.columns for col in spending_cols):
            # Total spending
            df['TotalSpent'] = df[spending_cols].sum(axis=1)

            # Has any spending
            df['HasSpending'] = (df['TotalSpent'] > 0).astype(int)

            # Number of services used
            df['NumServicesUsed'] = (df[spending_cols] > 0).sum(axis=1)

            # Average spending per service (excluding 0s)
            spending_array = df[spending_cols].values
            spending_array_nonzero = np.where(spending_array > 0, spending_array, np.nan)
            df['AvgSpendingPerService'] = np.nanmean(spending_array_nonzero, axis=1)
            df['AvgSpendingPerService'] = df['AvgSpendingPerService'].fillna(0)

        return df

    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age group categories."""
        if 'Age' in df.columns:
            # Create age bins
            df['AgeGroup'] = pd.cut(
                df['Age'],
                bins=[0, 12, 18, 25, 35, 50, 65, 100],
                labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAged', 'Senior', 'Elderly']
            )

            # Convert to string to allow 'Unknown' category
            df['AgeGroup'] = df['AgeGroup'].astype(str)
            df['AgeGroup'] = df['AgeGroup'].replace('nan', 'Unknown')

            # Also keep original Age (will be filled later)

        return df

    def _create_family_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Create family-related features."""
        if 'Group' in df.columns:
            # Family size (group size)
            if is_train:
                self.group_sizes = df.groupby('Group').size().to_dict()

            df['FamilySize'] = df['Group'].map(self.group_sizes)

            # Is traveling alone
            df['TravelingAlone'] = (df['FamilySize'] == 1).astype(int)

            # Is large family
            df['LargeFamily'] = (df['FamilySize'] >= 4).astype(int)

        # Extract surname from Name
        if 'Name' in df.columns:
            df['Surname'] = df['Name'].str.split().str[-1]

            # Count of same surname (family members)
            if is_train:
                self.surname_counts = df['Surname'].value_counts().to_dict()

            df['SurnameCount'] = df['Surname'].map(self.surname_counts).fillna(1)

            # Drop Name and Surname (too many unique values)
            df = df.drop(['Name', 'Surname'], axis=1)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently handle missing values based on data type."""
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical columns: fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)

        # Boolean columns: fill with False
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].fillna(False)

        return df

    def _log_transform_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to skewed spending columns."""
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']
        for col in spending_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].fillna(0))
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        # CryoSleep * HasSpending: cryo passengers shouldn't spend (data leak detector)
        if 'CryoSleep' in df.columns and 'HasSpending' in df.columns:
            cryo_numeric = df['CryoSleep'].astype(str).map({'True': 1, 'False': 0, 'nan': 0}).fillna(0)
            df['Cryo_x_Spending'] = cryo_numeric * df['HasSpending']

        # Age * TotalSpent: older passengers spend more?
        if 'Age' in df.columns and 'TotalSpent' in df.columns:
            df['Age_x_Spent'] = df['Age'].fillna(df['Age'].median()) * df['TotalSpent'].fillna(0)

        # VIP * TotalSpent: VIP spending amplifier
        if 'VIP' in df.columns and 'TotalSpent' in df.columns:
            vip_numeric = df['VIP'].astype(str).map({'True': 1, 'False': 0, 'nan': 0}).fillna(0)
            df['VIP_x_Spent'] = vip_numeric * df['TotalSpent'].fillna(0)

        # FamilySize * TotalSpent: bigger families spend differently
        if 'FamilySize' in df.columns and 'TotalSpent' in df.columns:
            df['Family_x_Spent'] = df['FamilySize'] * df['TotalSpent'].fillna(0)

        return df

    def _create_group_aggregates(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Group-level aggregates — strong signal in spaceship-titanic.

        Passengers in the same group tend to share fate (transported or not).
        Aggregating within-group stats exposes this pattern to tree splits.
        """
        if 'Group' not in df.columns:
            return df

        # Same-group spending stats (smoothing via combined fit at transform time
        # is OK here because test groups overlap with train groups in this dataset)
        if 'TotalSpent' in df.columns:
            df['GroupSpendMean'] = df.groupby('Group')['TotalSpent'].transform('mean')
            df['GroupSpendStd'] = df.groupby('Group')['TotalSpent'].transform('std').fillna(0)
            df['GroupSpendMax'] = df.groupby('Group')['TotalSpent'].transform('max')

        # Same-group cryo rate
        if 'CryoSleep' in df.columns:
            cryo_numeric = df['CryoSleep'].astype(str).map(
                {'True': 1.0, 'False': 0.0, 'nan': 0.5, 'NaN': 0.5}
            ).fillna(0.5).astype(float)
            df['GroupCryoRate'] = cryo_numeric.groupby(df['Group']).transform('mean')

        # Same-group age mean
        if 'Age' in df.columns:
            df['GroupAgeMean'] = df.groupby('Group')['Age'].transform('mean')

        # Group size already computed in family features as FamilySize; if not present,
        # expose it here for downstream interactions
        if 'FamilySize' not in df.columns:
            df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

        return df

    def _target_encode(self, df: pd.DataFrame, y: pd.Series | None, is_train: bool = True) -> pd.DataFrame:
        """Mean target encoding for high-cardinality categoricals.

        - On train: fit encoders from y, then apply
        - On test: apply saved encoders (y ignored)
        """
        target_cols = ['HomePlanet', 'Destination', 'CabinDeck']

        if is_train and y is not None:
            global_mean = float(y.mean())
            for col in target_cols:
                if col in df.columns:
                    grouped = y.groupby(df[col].fillna('Unknown')).agg(['mean', 'count'])
                    smoothing_weight = 10
                    grouped['smoothed'] = (
                        grouped['mean'] * grouped['count'] + global_mean * smoothing_weight
                    ) / (grouped['count'] + smoothing_weight)
                    self.target_encoders[col] = grouped['smoothed'].to_dict()
                    self.target_encoders[f'{col}_global'] = global_mean

        # Apply encoding (works for both train and test, as long as encoders are fit)
        for col in target_cols:
            if col in df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                global_mean = self.target_encoders.get(f'{col}_global', 0.5)
                df[f'{col}_target_enc'] = df[col].fillna('Unknown').map(encoder).fillna(global_mean)

        return df

    def _encode_categorical_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Handle unseen labels
                    df[col] = df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col].astype(str))

        return df


def create_feature_engineer(competition_id: str, flags: dict | None = None):
    """Factory function to create appropriate feature engineer.

    Args:
        competition_id: Competition name
        flags: Optional dict of feature engineering flags (gated per component)
    """
    if competition_id == "spaceship-titanic":
        return SpaceshipTitanicFeatureEngineer(flags=flags)
    else:
        return None


__all__ = ["SpaceshipTitanicFeatureEngineer", "create_feature_engineer"]
