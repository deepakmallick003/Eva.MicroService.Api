from typing import Optional
from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.security import OAuth2
from core.config import settings
from routers import health
from scripts import *

def get_application() -> FastAPI:

    class Oauth2ClientCredentials(OAuth2):
        def __init__(
            self,
            tokenUrl: str,
            client_id: str,
            scheme_name: str = None,
            scopes: dict = None,
            auto_error: bool = True,
        ):
            if not scopes:
                scopes = {}
            flows = OAuthFlowsModel(clientCredentials={"tokenUrl": tokenUrl, "scopes": scopes})
            self.client_id = client_id
            super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

        async def __call__(self, request: Request) -> Optional[str]:
            authorization: str = request.headers.get("Authorization")
            scheme, param = get_authorization_scheme_param(authorization)
            if not authorization or scheme.lower() != "bearer":
                if self.auto_error:
                    raise HTTPException(
                        status_code=401,
                        detail="Not authenticated",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    return None
            return param


    # Initialize the OAuth2 scheme for client credentials
    auth_scheme = Oauth2ClientCredentials(
        tokenUrl=f"https://login.microsoftonline.com/{settings.CABI_TenantId}/oauth2/v2.0/token",
        client_id=settings.EVA_API_ClientId,
        scopes={f'api://{settings.EVA_API_ClientId}/.default': 'api.read'}  
    )

    tags_metadata = [
        {
            "name": "EVA",
            "description": "**Utility** based API for **AI/ML/LLM** based development for different internal CABI use cases",
        }
    ]
    app = FastAPI(
        version='1.0.0',
        title=settings.PROJECT_NAME,
        description="**Utility** based API for **AI/ML/LLM** based development for different internal CABI use cases",
        docs_url=settings.DOC_URL,
        # root_path=settings.DEPLOYED_BASE_PATH,
        # openapi_tags=tags_metadata,      
    )

    # Enabled CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        # allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # include healthcheck router
    app.include_router(health.router)

    # redirect to swagger endpoint
    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        return RedirectResponse('swagger')


    @app.post("/chunk-data", response_model=model_chunk.ChunkDataResponse, 
              dependencies=[Security(auth_scheme)], tags=["API"])
    async def chunk_data(payload: model_chunk.ChunkDataRequest):
     
        try:
            chunk_processor = chunk.ChunkData(payload)
            chunked_data = chunk_processor.process_chunks()
            return chunked_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chunking operation failed: {str(e)}")

    
    # endpoint for data embed
    @app.post("/embed-chunks", response_model=model_embed.EmbedDataResponse, 
              dependencies=[Security(auth_scheme)], tags=["API"])
    async def embed_chunks(payload: model_embed.EmbedDataRequest):
        try:
            data_embedder = embed.EmbedData(payload)
            response = data_embedder.embed()
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding operation failed: {str(e)}")


    # endpoint for vector search
    @app.post("/vector-search", response_model=model_vector_search.VectorSearchResponse, 
              dependencies=[Security(auth_scheme)], tags=["API"])
    async def perform_vector_search(payload: model_vector_search.VectorSearchRequest):
        try:
            vector_searcher = vector_search.VectorSearch(payload)
            response = vector_searcher.search()
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")




    return app

app = get_application()
