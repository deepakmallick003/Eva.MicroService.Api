from scripts.models import model_vector_search
from scripts.vectordatabases import BaseDB

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from bson import ObjectId

class VectorSearch:
    def __init__(self, vector_search_data: model_vector_search.VectorSearchRequest):
        self.vs_data = vector_search_data

    def search(self):
        llm_embeddings = OpenAIEmbeddings(
            model = self.vs_data.llm_settings.embedding_model_name,
            api_key = self.vs_data.llm_settings.llm_key,
            dimensions= self.vs_data.llm_settings.vector_dimension_size                  
        )

        db_instance = BaseDB().get_vector_db(self.vs_data.db_type, self.vs_data.db_settings, llm_embeddings)

        result = db_instance.vector_search(self.vs_data.query)
        
        documents = []
        for doc, score in result:
            clean_metadata = {
                key: str(value) if not isinstance(value, str) and not isinstance(value, dict) else value
                for key, value in doc.metadata.items() if key != '_id' and key != 'embedding'
            }
            retrieved_chunk = model_vector_search.RetrievedDocumentChunk(
                content=doc.page_content,  
                metadata=clean_metadata,  
                relevancy_score=score    
            )
            
            documents.append(retrieved_chunk)

        return model_vector_search.VectorSearchResponse(documents=documents)