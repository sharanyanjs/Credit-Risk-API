from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = [
            'age', 'job', 'loan_amount', 'loan_duration',
            'debt_to_income', 'loan_to_value', 'payment_to_income'
        ]
    
    def fit(self, X):
        self.scaler.fit(X[[
            'loan_amount', 'loan_duration',
            'debt_to_income', 'loan_to_value', 'payment_to_income'
        ]])
    
    def transform(self, input_data: dict):
        """Transform raw input to model features"""
        # Calculate financial ratios
        monthly_payment = input_data['loan_amount'] / input_data['loan_duration']
        dti = (monthly_payment + input_data.get('existing_debt', 0)) / input_data['income']
        ltv = input_data['loan_amount'] / max(input_data.get('collateral_value', 1), 1)
        pti = monthly_payment / input_data['income']
        
        # Map employment type
        employment_map = {
            "Unemployed": 0,
            "Part-time": 1,
            "Full-time": 2,
            "Executive": 3
        }
        
        # Create feature dict
        features = {
            'age': input_data['age'],
            'job': employment_map.get(input_data['employment_type'], 2),
            'loan_amount': input_data['loan_amount'],
            'loan_duration': input_data['loan_duration'],
            'debt_to_income': dti,
            'loan_to_value': ltv,
            'payment_to_income': pti
        }
        
        # Scale features
        scaled_features = self.scaler.transform([[
            features['loan_amount'],
            features['loan_duration'],
            features['debt_to_income'],
            features['loan_to_value'],
            features['payment_to_income']
        ]])
        
        # Update scaled values
        features.update({
            'loan_amount': scaled_features[0][0],
            'loan_duration': scaled_features[0][1],
            'debt_to_income': scaled_features[0][2],
            'loan_to_value': scaled_features[0][3],
            'payment_to_income': scaled_features[0][4]
        })
        
        return features
