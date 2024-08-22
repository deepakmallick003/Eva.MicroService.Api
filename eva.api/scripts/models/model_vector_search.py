from .model_base import *

class VectorSearchRequest(BaseModel):
    db_type: VectorDatabaseType 
    db_settings: DBSettings 
    llm_settings: LLMSettings
    query: str 

class VectorSearchResponse(BaseModel):
     documents: Optional[List[RetrievedDocumentChunk]] = None 