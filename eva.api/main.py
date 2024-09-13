from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.auth import auth_scheme
from routers import health
from scripts import *

def get_application() -> FastAPI:

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
        root_path=settings.DEPLOYED_BASE_PATH
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


    # endpoint for chatbot response
    @app.post("/chatbot-response", response_model=model_rag.ChatResponse, 
              dependencies=[Security(auth_scheme)], tags=["API"])
    async def get_chatbot_response(payload: model_rag.ChatRequest):
        try:
            chat_processor = rag.RAG(payload)
            response = chat_processor.get_response()
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chatbot request failed: {str(e)}")



    return app

app = get_application()
