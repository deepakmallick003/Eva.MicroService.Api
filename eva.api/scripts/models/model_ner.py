from .model_base import *

class Concept(BaseModel):
    concept_name: str
    concept_source: str

class NERModelRequest(BaseModel):
    project_directory_name: str = Field(
        ..., 
        description="Directory name for saving or loading the NER model."
    )
    concepts: Optional[List[Concept]] = Field(
        None, 
        description="A 'Concepts' object containing concept names and their sources. Required for training."
    )
