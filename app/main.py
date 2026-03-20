import logging

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.api.routes.claims import router as claims_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InsurFlow AI", version="0.1.0")

app.include_router(health_router)
app.include_router(inference_router)
app.include_router(claims_router)


@app.on_event("startup")
async def _startup() -> None:
    # Place for bootstrapping global resources (e.g., model clients, DB pools).
    logger.info("Starting InsurFlow AI")

