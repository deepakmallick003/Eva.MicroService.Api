from scripts.models import model_embed
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from scripts.vectordatabases import BaseDB

class EmbedData:
    def __init__(self, embed_data: model_embed.EmbedDataRequest):
        self.embed_data = embed_data

    def embed(self):
        llm_embeddings = OpenAIEmbeddings(
            model = self.embed_data.llm_settings.embedding_model_name,
            api_key = self.embed_data.llm_settings.llm_key,
            dimensions= self.embed_data.llm_settings.vector_dimension_size                  
        )

        db_instance = BaseDB().get_vector_db(self.embed_data.db_type)

        result_message = db_instance.embed_documents(llm_embeddings, self.embed_data)
        
        return model_embed.EmbedDataResponse(success=True, message=result_message) 