from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.services.risk import RiskAssessmentService
from app.ml.model import load_model
from app.ml.preprocessor import Preprocessor

router = APIRouter()

class CreditApplication(BaseModel):
    age: int
    employment_type: str
    income: float
    loan_amount: float
    loan_duration: int
    existing_debt: float = 0
    collateral_value: float = 0
    credit_score: Optional[int] = None
    recent_inquiries: int = 0
    is_existing_client: bool = False
    is_preferred_client: bool = False

@router.post("/assess")
async def assess_risk(application: CreditApplication):
    try:
        model = load_model()
        preprocessor = Preprocessor()
        
        # Transform input
        processed_data = preprocessor.transform(application.dict())
        
        # Get prediction
        service = RiskAssessmentService(model)
        result = service.assess(processed_data)
        
        return {
            "risk_score": result.risk_score,
            "risk_category": result.risk_category,
            "expected_profit": result.expected_profit,
            "recommendation": result.recommendation,
            "pricing_adjustment": result.pricing_adjustment,
            "key_factors": result.key_factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
