from pydantic import BaseModel, field_validator, Field, model_validator
from typing import List, Dict, Optional, Any, Union
from typing_extensions import Annotated
from enum import Enum


# Enums #

class VectorDatabaseType(str, Enum):
    """Defines the types of vector databases supported."""
    MongoDB = 'mongodb'
    Neo4j = 'neo4j'
    Weaviate = 'weviate'

class VectorSimilarityFunction(str, Enum):
    """Available similarity functions for vector search."""
    Cosine = 'cosine'
    Euclidean = 'euclidean'
    DotProduct = 'dotProduct'

class RetrieverSearchType(str, Enum):
    """Available similarity functions for vector search."""
    Similarity = 'similarity'
    Similarity_Score_Threshold = 'similarity_score_threshold'
    MMR = 'mmr'

class EmbeddingModelName(str, Enum):
    """Supported embedding models."""
    TextEmbedding3Small = 'text-embedding-3-small'
    TextEmbedding3Large = 'text-embedding-3-large'
    TextEmbeddingAda002 = 'text-embedding-ada-002'

class VectorDimensionSize(int, Enum):
    """Supported vector dimension sizes."""
    Dim256 = 256
    Dim512 = 512
    Dim1024 = 1024
    Dim1536 = 1536
    Dim3072 = 3072

class ChatModelName(str, Enum):
    """Supported ChatGPT models."""
    GPT_3_5_Turbo = 'gpt-3.5-turbo'
    GPT_3_5_Turbo_16K = 'gpt-3.5-turbo-16k'
    GPT_4 = 'gpt-4'
    GPT_4_32K = 'gpt-4-turbo'
    GPT_4_Omni = 'gpt-4o'

class MaxChunkLimit(int, Enum):
    """Maximum chunks to retrieve for the response."""
    Limit_5 = 5
    Limit_10 = 10
    Limit_20 = 20
    Limit_50 = 50

class ChunkRelevance(float, Enum):
    """Chunk relevance score threshold."""
    Percent_10 = 0.1
    Percent_15 = 0.15
    Percent_20 = 0.2
    Percent_25 = 0.25
    Percent_30 = 0.3
    Percent_35 = 0.35
    Percent_40 = 0.4
    Percent_45 = 0.45
    Percent_50 = 0.5
    Percent_55 = 0.55
    Percent_60 = 0.6
    Percent_65 = 0.65
    Percent_70 = 0.7
    Percent_75 = 0.75
    Percent_80 = 0.8
    Percent_85 = 0.85
    Percent_90 = 0.9
    Percent_95 = 0.95
    Percent_100 = 1.0

class TemperatureLevel(float, Enum):
    """Temperature settings for the chat model."""
    VeryLow = 0.0    # Completely deterministic, no randomness.
    Low = 0.1        # Minimal randomness, very focused and precise.
    MediumLow = 0.2  # Slight randomness, but still precise.
    Medium = 0.3     # Balanced randomness, allows for some creativity.
    MediumHigh = 0.5 # More creativity, still relatively controlled.
    High = 0.7       # Significant randomness, creative responses.
    VeryHigh = 1.0   # Maximum randomness, highly creative and varied responses.

class FrequencyPenalty(float, Enum):
    """Frequency penalty to discourage repeating the same tokens."""
    NoPenalty = 0.0    # No penalty, allows for repeated words.
    Low = 0.1          # Slight penalty for repetition.
    MediumLow = 0.2    # Moderate penalty for repetition.
    Medium = 0.5       # Balanced penalty, noticeable discouragement of repetition.
    High = 0.7         # Significant penalty, strong discouragement of repetition.
    VeryHigh = 1.0     # Maximum penalty, repetition is highly discouraged.

class PresencePenalty(float, Enum):
    """Presence penalty to encourage or discourage introducing new topics."""
    NoPenalty = 0.0    # No penalty, allows for new or unrelated topics.
    Low = 0.1          # Slight penalty, encourages staying somewhat on topic.
    MediumLow = 0.2    # Moderate penalty, discourages introducing unrelated topics.
    Medium = 0.5       # Balanced penalty, noticeably discourages new topics.
    High = 0.7         # Strong penalty, highly discourages new topics.
    VeryHigh = 1.0     # Maximum penalty, sticks strictly to the current topic.

class RAGRoles(str, Enum):
    """Role in a rag converstaion can be either 'Human' or 'AI'"""
    Human = "Human"
    AI = "AI"    

class RAGStrategy(str, Enum):
    """Different Types of RAG Strategy"""
    Version1 = "v1"
    Version2 = "v2"
    Version3 = "v3"

######


class DocumentChunk(BaseModel):
    """Represents a document with chunked text and associated metadata."""
    content: str
    metadata: Dict[str, str]

    @field_validator('metadata')
    def check_required_keys(cls, v):
        required_keys = ['source', 'type']
        if v is None:
            raise ValueError("metadata cannot be None and must contain 'source' and 'type'.")
        for key in required_keys:
            if key not in v:
                raise ValueError(f"'{key}' is a required key in metadata.")
        return v

class RetrievedDocumentChunk(BaseModel):
    """Represents a document retrieved from the vector search, with a relevancy score."""
    content: str
    metadata: Dict[str, Any]
    relevancy_score: float

class DBSettings(BaseModel):
    """
    Database settings for connecting to the vector database.
    """
    uri: str
    db_name: str
    collection_name: str
    vector_index_name: Optional[str] = None
    vector_similarity_function: VectorSimilarityFunction = VectorSimilarityFunction.Cosine

class LLMSettings(BaseModel):
    """
    LLM configuration settings, including the embedding model and vector dimension size.
    """
    llm_key: str
    vector_dimension_size: VectorDimensionSize = VectorDimensionSize.Dim1536
    embedding_model_name: EmbeddingModelName = EmbeddingModelName.TextEmbedding3Small

class RetrieverSettings(BaseModel):
    search_type: RetrieverSearchType = RetrieverSearchType.MMR
    max_chunks_to_retrieve: Annotated[int, Field(ge=10, le=50)] = 10  
    retrieved_chunks_min_relevance_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    fetch_k: Annotated[int, Field(ge=50, le=1000)] = 200  
    lambda_mult: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3

class RAGSettings(BaseModel):
    """
    RAG (Retrieval-Augmented Generation) configuration settings.
    """
    chat_model_name: Optional[ChatModelName] = ChatModelName.GPT_4_Omni
    max_tokens_for_response: Annotated[int, Field(ge=50, le=2000)] = 500  
    retriever_search_settings: RetrieverSettings = RetrieverSettings()
    temperature: Optional[TemperatureLevel] = TemperatureLevel.VeryLow
    frequency_penalty: Optional[FrequencyPenalty] = FrequencyPenalty.MediumLow
    presence_penalty: Optional[PresencePenalty] = PresencePenalty.MediumLow
    max_tokens_for_history: Annotated[int, Field(ge=50, le=2000)] = 300 
    streaming_response: bool = False
