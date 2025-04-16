from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from app.ml.explainer import SHAPExplainer

@dataclass
class RiskAssessmentResult:
    risk_score: float
    risk_category: str
    expected_profit: float
    recommendation: str
    pricing_adjustment: str
    key_factors: List[Dict[str, float]]

class RiskAssessmentService:
    def __init__(self, model):
        self.model = model
        self.explainer = SHAPExplainer(model)
        
        # Business rules
        self.risk_categories = {
            "Low": (0, 0.3),
            "Medium": (0.3, 0.7),
            "High": (0.7, 1.0)
        }
    
    def assess(self, input_data):
        # Get prediction
        proba = self.model.predict_proba([input_data])[0][1]
        risk_score = round(proba * 100, 1)
        
        # Determine risk category
        risk_category = next(
            (k for k, (low, high) in self.risk_categories.items() 
            if low <= proba < high), 
            "High"
        )
        
        # Business calculations
        expected_profit = self._calculate_expected_profit(
            input_data['loan_amount'],
            proba,
            input_data.get('is_existing_client', False)
        )
        
        # Generate explanation
        shap_values = self.explainer.explain(input_data)
        key_factors = [
            {"feature": k, "impact": float(v)} 
            for k, v in shap_values.items()
        ]
        
        # Generate recommendations
        recommendation, pricing = self._generate_recommendation(
            risk_category, 
            proba,
            input_data.get('collateral_value', 0)
        )
        
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_category=risk_category,
            expected_profit=expected_profit,
            recommendation=recommendation,
            pricing_adjustment=pricing,
            key_factors=key_factors
        )
    
    def _calculate_expected_profit(self, amount, risk_prob, is_existing_client):
        base_rate = 0.08
        adj_rate = base_rate + (0.01 if is_existing_client else 0)
        return (amount * adj_rate) - (amount * risk_prob * 0.85)
    
    def _generate_recommendation(self, risk_category, proba, collateral):
        if risk_category == "High":
            if collateral > 0:
                return "Approve with collateral", "+400 bps"
            return "Decline application", "N/A"
        elif risk_category == "Medium":
            return "Approve with conditions", f"+{int((proba-0.3)*200)} bps"
        return "Approve", "Standard rate"
