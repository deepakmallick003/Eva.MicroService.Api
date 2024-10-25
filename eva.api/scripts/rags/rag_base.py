from .rag_v1 import RAG_V1
from .rag_v2 import RAG_V2
from .rag_v3 import RAG_V3

from scripts.models import model_rag

class BaseRAG():
    def __init__(self):
        pass

    def get_rag(self, rag_type: model_rag.RAGStrategy, chat_data: model_rag.ChatRequest):
        if rag_type == model_rag.RAGStrategy.Version2:
            return RAG_V2(chat_data)
        elif rag_type == model_rag.RAGStrategy.Version3:
            return RAG_V3(chat_data)
        else:
            #Default
            return RAG_V1(chat_data)


