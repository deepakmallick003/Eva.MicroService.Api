from .model_base import *
from .model_ner import NERModelRequest

class EmbedDataRequest(BaseModel):
    db_type: VectorDatabaseType 
    db_settings: DBSettings
    llm_settings: LLMSettings 
    documents: List[DocumentChunk]
    train_ner_for_concepts: bool = Field(
        default=False, 
        description=(
            "If true, the system will train/retrain the NER model using the provided 'concepts'. "
            "'project_directory_name' and 'concepts' must be provided when this is set to true."
        )
    )
    ner_model: Optional[NERModelRequest] = Field(
        None,
        description="Details required to train or load the NER model, including concepts and project directory."
    )

class EmbedDataResponse(BaseModel):
    success: bool  # Indicates if the operation was successful or not
    message: str   # Contains a message about the operation