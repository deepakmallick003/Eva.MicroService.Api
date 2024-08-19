from .model_base import *

class DataChunks(BaseModel):
    data_source_type: str 
    property_meta_data_map: Optional[Dict[str, str]] = None
    content: Optional[Dict[str, Any]] = None

class ChunkDataRequest(BaseModel):
    chunks: List[DataChunks]

class ChunkDataResponse(BaseModel):
    documents: List[DocumentChunk] 