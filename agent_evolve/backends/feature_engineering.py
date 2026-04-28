"""Advanced feature engineering for Spaceship Titanic.

This module provides sophisticated feature engineering that goes beyond
simple fillna + LabelEncoder to extract meaningful patterns from the data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class SpaceshipTitanicFeatureEngineer:
    """Feature engineering specifically for Spaceship Titanic competition."""

    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Transform features for training set."""
        df = df.copy()

        # 1. Extract PassengerId features
        df = self._extract_passenger_id_features(df)

        # 2. Extract Cabin features
        df = self._extract_cabin_features(df)

        # 3. Create spending features
        df = self._create_spending_features(df)

        # 4. Create age groups
        df = self._create_age_groups(df)

        # 5. Create family features
        df = self._create_family_features(df, is_train=is_train)

        # 6. Handle missing values intelligently
        df = self._handle_missing_values(df)

        # 7. Encode categorical features
        df = self._encode_categorical_features(df, is_train=is_train)

        if is_train:
            self.feature_names = df.columns.tolist()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features for test set."""
        return self.fit_transform(df, is_train=False)

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


def create_feature_engineer(competition_id: str):
    """Factory function to create appropriate feature engineer."""
    if competition_id == "spaceship-titanic":
        return SpaceshipTitanicFeatureEngineer()
    else:
        # Fallback to basic feature engineering
        return None


__all__ = ["SpaceshipTitanicFeatureEngineer", "create_feature_engineer"]
