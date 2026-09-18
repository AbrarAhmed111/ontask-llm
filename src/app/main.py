"""
FastAPI Application Entrypoint.
Initializes FastAPI, configures CORS, and mounts API routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.core.config import get_settings
from src.app.core.logging import setup_logging
from src.app.api.router import api_router
from src.app.api.routes.health import router as health_router

settings = get_settings()
setup_logging(settings.LOG_LEVEL)

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "OnTask's AI microservice: turns an already-computed structured snapshot of a "
        "workspace's rolling-24h focus-time activity into a validated, grounded narrative "
        "(the automatic Daily Report), using a multi-provider LLM Gateway with automatic "
        "failover."
    ),
)

# -----------------------------------------------------------------------------
# CORS Middleware
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount root health check (e.g. for container health checks)
app.include_router(health_router)

# Mount central API router under /api
app.include_router(api_router)


@app.get("/", summary="Root index")
async def root():
    """Returns basic service status and documentation link."""
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "generate_summary": "/api/summary/generate",
            "health": "/health",
        },
    }
