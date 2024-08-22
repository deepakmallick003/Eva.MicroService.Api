# Module for exposing healthchecks endpoints
# these endpoints are used by k8

from fastapi import APIRouter

router = APIRouter(tags=["health Check"])

@router.get("/health", include_in_schema=False)
async def health():
    return {"message": "ok"}

@router.get("/health/kubernetes", include_in_schema=False)
async def healthkubernetes():
    return {"message": "ok"}