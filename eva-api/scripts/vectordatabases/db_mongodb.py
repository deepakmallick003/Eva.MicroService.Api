# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.vectorstores import MongoDBAtlasVectorSearch

class MongoDB():
    def __init__(self):
        pass

    def get_vector_index(self, data):
        # Extract settings from db_settings
        uri = data.db_settings.uri
        dbname = data.db_settings.db_name
        collectionname = data.db_settings.collection_name
        indexname = data.db_settings.vector_index_name

        # Initialize the vector search instance
        vector_index = MongoDBAtlasVectorSearch.from_connection_string(
            uri,
            dbname + "." + collectionname,
            embedding = self.llm_embeddings,
            index_name = indexname
        )

    def embed_documents(self, llm_embeddings, embed_data):
        # Get Vector Index
        self.llm_embeddings = llm_embeddings
        self.vector_index = self.get_vector_index(embed_data)
        
        # Add documents to the vector store
        self.vector_index.add_documents(documents= embed_data.documents)
        
        return f"Documents successfully embedded in MongoDB: {embed_data.db_settings.collectionname}"
    

    def vector_search(self, llm_embeddings, vs_data):
        # Get Vector Index
        self.llm_embeddings = llm_embeddings
        self.vector_index = self.get_vector_index(vs_data)

        # Add documents to the vector store
        results = self.vector_index.similarity_search(
            query = vs_data.query,
            k = 10,
        )
        
        return results
