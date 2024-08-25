from .model_base import *

class DataChunks(BaseModel):
    data_source_type: str 
    property_meta_data_map: Optional[Dict[str, str]] = None
    content: Optional[Dict[str, Any]] = None

    @field_validator('property_meta_data_map')
    def check_required_keys(cls, v):
        required_keys = ['Source', 'Type']
        if v is None:
            raise ValueError("property_meta_data_map cannot be None and must contain 'Source' and 'Type'.")
        for key in required_keys:
            if key not in v:
                raise ValueError(f"'{key}' is a required key in property_meta_data_map.")
        return v

class ChunkDataRequest(BaseModel):
    chunks: List[DataChunks]

class ChunkDataResponse(BaseModel):
    documents: List[DocumentChunk] 