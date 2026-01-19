import joblib
import numpy as np
import pandas as pd
from .preprocessor import CreditDataPreprocessor

class CreditScoringPredictor:
    def __init__(self, model_path='models/trained_model.pkl'):
        self.model = None
        self.preprocessor = CreditDataPreprocessor()
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model and preprocessor"""
        try:
            self.model = joblib.load(model_path)
            self.preprocessor.load_preprocessor('models/preprocessor.pkl')
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Convert input data to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input data must be a dictionary or DataFrame")
        
        # Preprocess features
        processed_data = self.preprocessor.preprocess_features(input_df, fit=False)
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(
                zip(processed_data.columns, self.model.feature_importances_)
            )
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'class_probabilities': probability[0].tolist(),
            'feature_importance': feature_importance,
            'credit_rating': self.get_credit_rating(probability[0][1])
        }
    
    def get_credit_rating(self, probability):
        """Convert probability to credit rating"""
        if probability >= 0.9:
            return 'Excellent (A+)'
        elif probability >= 0.8:
            return 'Very Good (A)'
        elif probability >= 0.7:
            return 'Good (B)'
        elif probability >= 0.6:
            return 'Fair (C)'
        elif probability >= 0.4:
            return 'Poor (D)'
        else:
            return 'Very Poor (E)'
    
    def batch_predict(self, input_df):
        """Make predictions on batch data"""
        predictions = []
        for _, row in input_df.iterrows():
            prediction = self.predict(row.to_dict())
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)