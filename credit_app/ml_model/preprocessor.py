import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class CreditDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_features(self, X, fit=False):
        """Preprocess credit scoring features"""
        
        # Create a copy to avoid modifying original
        X_processed = X.copy()
        
        # Handle categorical features
        categorical_cols = ['employment_status', 'education_level', 'home_ownership', 'loan_purpose']
        for col in categorical_cols:
            if col in X_processed.columns:
                if fit:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        X_processed[col] = X_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

        # Calculate derived features first
        if 'annual_income' in X_processed.columns and 'monthly_debt' in X_processed.columns:
            X_processed['debt_to_income_ratio'] = (
                X_processed['monthly_debt'] * 12 / X_processed['annual_income']
            )

        if 'credit_card_limit' in X_processed.columns and 'credit_card_balance' in X_processed.columns:
            X_processed['credit_utilization_ratio'] = X_processed['credit_card_balance'] / X_processed['credit_card_limit']

        if 'credit_utilization_ratio' in X_processed.columns:
            X_processed['credit_utilization_category'] = pd.cut(
                X_processed['credit_utilization_ratio'],
                bins=[0, 0.3, 0.7, 1.0, float('inf')],
                labels=[0, 1, 2, 3]
            )

        # Additional feature engineering
        if 'number_of_loans' in X_processed.columns:
            X_processed['has_multiple_loans'] = (X_processed['number_of_loans'] > 1).astype(int)

        # Payment history score
        payment_cols = [col for col in X_processed.columns if 'late_payments' in col]
        if payment_cols:
            X_processed['payment_score'] = X_processed[payment_cols].sum(axis=1)

        # Numerical features that need scaling
        numerical_cols = [
            'age', 'annual_income', 'monthly_debt', 'credit_utilization_ratio',
            'number_of_credit_lines', 'loan_amount',
            'debt_to_income_ratio', 'loan_term_months', 'late_payments_30days',
            'late_payments_60days', 'late_payments_90days', 'credit_card_limit',
            'credit_card_balance', 'has_bank_account', 'number_of_loans', 'months_employed'
        ]

        # Filter numerical columns that exist in the dataframe
        existing_numerical_cols = [col for col in numerical_cols if col in X_processed.columns]

        # Impute missing values
        if fit:
            X_processed[existing_numerical_cols] = self.imputer.fit_transform(X_processed[existing_numerical_cols])
        else:
            X_processed[existing_numerical_cols] = self.imputer.transform(X_processed[existing_numerical_cols])

        # Scale numerical features
        if fit:
            X_processed[existing_numerical_cols] = self.scaler.fit_transform(X_processed[existing_numerical_cols])
            self.feature_names = X_processed.columns.tolist()
        else:
            X_processed[existing_numerical_cols] = self.scaler.transform(X_processed[existing_numerical_cols])
            # Reorder columns to match training order
            if self.feature_names:
                X_processed = X_processed[self.feature_names]

        return X_processed
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save the preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)

    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load the preprocessor"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.label_encoders = data['label_encoders']
        self.feature_names = data.get('feature_names')
