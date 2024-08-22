from .model_base import *

class EmbedDataRequest(BaseModel):
    db_type: VectorDatabaseType 
    db_settings: DBSettings
    llm_settings: LLMSettings 
    documents: List[DocumentChunk]

class EmbedDataResponse(BaseModel):
    success: bool  # Indicates if the operation was successful or not
    message: str   # Contains a message about the operation