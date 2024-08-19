from .models import model_vector_search
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from .vectordatabases import BaseDB

class VectorSearch:
    def __init__(self, vector_search_data: model_vector_search.VectorSearchRequest):
        self.vs_data = vector_search_data

    def search(self):
        llm_embeddings = OpenAIEmbeddings(
            model = self.vs_data.llm_settings.embedding_model_name,
            api_key = self.vs_data.llm_settings.llm_key,
            dimensions= self.vs_data.llm_settings.vector_dimension_size                  
        )

        db_instance = BaseDB().get_vector_db(self.vs_data.db_type)

        result = db_instance.vector_search(llm_embeddings, self.vs_data)
        
        return result