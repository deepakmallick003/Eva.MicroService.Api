from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum


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

class DocumentChunk(BaseModel):
    """Represents a document with chunked text and associated metadata."""
    content: str
    metadata: Dict[str, str]

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
