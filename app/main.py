from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import assessment, portfolio, model
from app.core.config import settings

app = FastAPI(
    title="Goldman Sachs Credit Risk API",
    description="Enterprise credit risk assessment platform",
    version="3.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    assessment.router,
    prefix="/api/v1/assessment",
    tags=["Risk Assessment"]
)
app.include_router(
    portfolio.router,
    prefix="/api/v1/portfolio",
    tags=["Portfolio Analytics"]
)
app.include_router(
    model.router,
    prefix="/api/v1/model",
    tags=["Model Management"]
)

@app.get("/")
async def root():
    return {"message": "GS Credit Risk API"}
