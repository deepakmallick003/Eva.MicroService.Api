from scripts.models import model_embed
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

        db_instance = BaseDB().get_vector_db(self.embed_data.db_type, self.embed_data.db_settings, llm_embeddings)
        result_message = db_instance.embed_documents(self.embed_data.documents)

        self._train_ner_model_if_required()
        
        return model_embed.EmbedDataResponse(success=True, message=result_message)
    
    def _train_ner_model_if_required(self):
        if self.embed_data.train_ner_for_concepts and self.embed_data.ner_model:
            from scripts.ner import Concept_NER
            from scripts.models import model_ner 

            ner_data = model_ner.NERModelRequest(
                project_directory_name=self.embed_data.ner_model.project_directory_name,
                concepts=self.embed_data.ner_model.concepts
            )

            concept_ner = Concept_NER(ner_data)
            concept_ner.train_concepts_ner_model()

            return "NER model successfully trained with provided concepts."
        
        return None