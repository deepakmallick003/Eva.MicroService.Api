from pymongo import MongoClient
from collections import defaultdict
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document
from datetime import datetime, timezone
import hashlib

class MongoDB():
    def __init__(self, db_settings=None, llm_embeddings=None):
        if db_settings:
            self.db_settings = db_settings
            self.client = MongoClient(db_settings.uri)
            self.db = self.client[db_settings.db_name]
            self.collection = self.db[db_settings.collection_name]

        if llm_embeddings:
            self.llm_embeddings = llm_embeddings
            self.vector_index = self._get_vector_index()

    ##Public Methdos

    def embed_documents(self, documents):
        """
        Public method to embed or update the documents in the vector store. 
        Group all chunks of the same document together before deleting and re-embedding.
        """

        # Group documents by source
        grouped_chunks = self._group_chunks_by_source_document(documents)

        for source, chunk_group in grouped_chunks.items():
            # Check if any chunks with the same document source exist and delete if they do
            self._check_and_delete_existing_chunk_groups(source)

            new_chunks = self._prepare_new_chunks(chunk_group, source)

            # Add the new embeddings to the vector store
            self._add_chunks_to_vector_store(new_chunks)

        return  f"Documents successfully embedded in MongoDB: {self.db_settings.collection_name}"

    def vector_search(self, query):
        """
        Public method to perform vector search on the embedded documents.
        """
        # Perform similarity search with relevance scores
        results = self.vector_index.similarity_search_with_relevance_scores(
            query = query,
            k = 10,
        )
        
        return results

    ##Public Methdos

    def _get_vector_index(self):
        """
        Private method to initialize the vector index from the MongoDB Atlas connection.
        """
        # Initialize the vector search instance
        vector_index = MongoDBAtlasVectorSearch.from_connection_string(
            self.db_settings.uri,
            self.db_settings.db_name + "." + self.db_settings.collection_name,
            embedding=self.llm_embeddings,
            index_name=self.db_settings.vector_index_name
        )
        return vector_index

    def _group_chunks_by_source_document(self, chunks):
        """
        Private method to group document chunks by their source.
        """
        grouped_chunks = defaultdict(list)
        for data in chunks:
            grouped_chunks[data.metadata['Source']].append(data)
        return grouped_chunks

    def _check_and_delete_existing_chunk_groups(self, source):
        """
        Private method to check if any documents with the given source exist in the MongoDB collection,
        and delete them if they do.
        """
        # Use count_documents to check if there are any documents with the given source
        doc_count = self.collection.count_documents({"Source": source})

        if doc_count > 0:
            # If documents exist, delete them
            self.collection.delete_many({"Source": source})
            return True

        return False

    def _prepare_new_chunks(self, chunk_group, source):
        """
        Private method to prepare new embeddings for all chunks of the document,
        generate content hashes for deduplication, and use the hash as document_id.
        """
        new_chunks = []
        last_updated = datetime.now(timezone.utc).isoformat() # Timezone-aware timestamp

        # Track unique content hashes within the same chunk group
        seen_hashes = set()

        for data in chunk_group:
            # Generate a hash of the chunk content, source, and source_type
            content_hash = self._generate_content_hash(data.content, data.metadata['Source'], data.metadata['Type'])

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)

                # Use the content_hash as the document_id
                new_chunks.append(
                    Document(
                        id=content_hash,  # Assigning the content hash as the document_id
                        page_content=data.content,
                        metadata={
                            "_id": content_hash,
                            "Last_Updated": last_updated,  # Timestamp for versioning
                            **data.metadata  # Include other metadata
                        }
                    )
                )

        return new_chunks

    def _generate_content_hash(self, content, source, type):
        """
        Private method to generate a SHA256 hash of the combination of content, source, and type for deduplication.
        """
        combined_data = f"{source}-{type}-{content}"
        return hashlib.sha256(combined_data.encode('utf-8')).hexdigest()

    def _add_chunks_to_vector_store(self, new_chunks):
        """
        Private method to add new chunks to the vector store.
        """
        self.vector_index.add_documents(documents=new_chunks)
