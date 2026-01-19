import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
from .preprocessor import CreditDataPreprocessor

class CreditModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        self.best_model = None
        self.preprocessor = CreditDataPreprocessor()
        self.results = {}
        
    def load_and_prepare_data(self, data_path='data/credit_dataset.csv'):
        """Load and prepare the credit scoring dataset"""
        
        # Load dataset
        df = pd.read_csv(data_path)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Define features and target
        X = df.drop('credit_worthiness', axis=1)
        y = df['credit_worthiness']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess features before SMOTE
        X_train_processed = self.preprocessor.preprocess_features(
            X_train, fit=True
        )
        X_test_processed = self.preprocessor.preprocess_features(
            X_test, fit=False
        )

        # Skip SMOTE for now due to small dataset and categorical encoding issues
        return X_train_processed, X_test_processed, y_train, y_test
    
    def feature_engineering(self, df):
        """Perform feature engineering on credit data"""
        
        # Create new features
        if 'annual_income' in df.columns and 'monthly_debt' in df.columns:
            df['debt_to_income_ratio'] = (df['monthly_debt'] * 12) / df['annual_income']
        
        if 'credit_card_limit' in df.columns and 'credit_card_balance' in df.columns:
            df['credit_utilization_ratio'] = df['credit_card_balance'] / df['credit_card_limit']
        
        if 'number_of_loans' in df.columns:
            df['has_multiple_loans'] = (df['number_of_loans'] > 1).astype(int)
        
        # Payment history score
        payment_cols = [col for col in df.columns if 'payment' in col.lower()]
        if payment_cols:
            df['payment_score'] = df[payment_cols].mean(axis=1)
        
        return df
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate multiple models"""
        
        model_results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'model': model
            }
            
            model_results[model_name] = metrics
            
            # Print results
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.results = model_results
        
        # Select best model based on ROC-AUC
        best_model_name = max(model_results, key=lambda x: model_results[x]['roc_auc'])
        self.best_model = model_results[best_model_name]['model']
        
        return model_results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest"""
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def save_model(self, path='models/trained_model.pkl'):
        """Save the trained model and preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, path)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        # Save training results
        results_path = path.replace('.pkl', '_results.pkl')
        joblib.dump(self.results, results_path)
        
        print(f"Model saved to {path}")
        print(f"Preprocessor saved to models/preprocessor.pkl")
    
    def generate_evaluation_plots(self, X_test, y_test, output_dir='static/plots'):
        """Generate evaluation plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{output_dir}/confusion_matrix.png')
        plt.close()
        
        # ROC Curve
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'{output_dir}/roc_curve.png')
        plt.close()
        
        # Feature Importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': range(len(self.best_model.feature_importances_)),
                'importance': self.best_model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'][:20], feature_importance['importance'][:20])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.savefig(f'{output_dir}/feature_importance.png')
            plt.close()