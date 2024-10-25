from scripts.models import model_rag
from scripts.rags import BaseRAG

class RAG:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data

    ##Public Methods

    def get_response(self):
        rag_instance = BaseRAG().get_rag(self.chat_data.rag_type, self.chat_data)
        result = rag_instance.get_response()
        return result
    
    