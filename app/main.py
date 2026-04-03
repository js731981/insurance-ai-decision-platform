import logging
import os

# Chroma reads this at import time; avoids broken PostHog telemetry spam in local dev.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logging.basicConfig(level=logging.INFO)
for _telemetry in ("chromadb.telemetry", "chromadb.telemetry.product.posthog"):
    logging.getLogger(_telemetry).setLevel(logging.CRITICAL)

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.analytics import router as analytics_router
from app.api.routes.cases import router as cases_router
from app.api.routes.health import router as health_router
from app.api.routes.claims import router as claims_router
from app.api.routes.inference import router as inference_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="InsurFlow AI",
    version="0.1.0",
    description="API for micro-insurance: fast, automated triage of small-ticket claims.",
)

_web_dir = Path(__file__).resolve().parent / "web"
app.mount("/ui", StaticFiles(directory=str(_web_dir), html=True), name="ui")

app.include_router(health_router)
app.include_router(inference_router)
app.include_router(claims_router)
app.include_router(cases_router)
app.include_router(analytics_router)


@app.on_event("startup")
async def _startup() -> None:
    logger.info("Starting InsurFlow AI (micro-insurance)")
