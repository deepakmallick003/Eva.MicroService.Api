from .db_mongodb import MongoDB
from .db_neo4j import Neo4J
from .db_weaviate import Weaviate

class BaseDB():
    def __init__(self):
        pass

    def get_vector_db(self, db_type, db_settings, llm_embeddings):
        if db_type == db_type.MongoDB:
            return MongoDB(db_settings, llm_embeddings)
        elif db_type == db_type.Neo4j:
            return Neo4J()
        elif db_type == db_type.Weaviate:
            return Weaviate()


