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
        # Group documents by all metadata key-value pairs
        grouped_chunks = self._group_chunks_by_all_metadata(documents)

        # Check and delete existing chunks based on the groupings
        self._check_and_delete_existing_chunk_groups(grouped_chunks)

        # Prepare and embed new chunks
        for metadata_key, chunk_group in grouped_chunks.items():
            new_chunks = self._prepare_new_chunks(chunk_group)
            self._add_chunks_to_vector_store(new_chunks)

        return f"Documents successfully embedded in MongoDB: {self.db_settings.collection_name}"


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

    def _group_chunks_by_all_metadata(self, chunks):
        """
        Private method to group document chunks by all metadata key-value pairs.
        """
        grouped_chunks = defaultdict(list)

        for data in chunks:
            # Create a combined key for all metadata values
            metadata_key = tuple(sorted((k, v) for k, v in data.metadata.items()))

            # Group chunks by this combined metadata key
            grouped_chunks[metadata_key].append(data)

        return grouped_chunks

    def _check_and_delete_existing_chunk_groups(self, grouped_chunks):
        """
        Private method to check if any documents with the given metadata exist in the MongoDB collection,
        and delete them if they do.
        """
        for metadata_key, chunk_group in grouped_chunks.items():
            # Convert the metadata_key tuple back to a dictionary for the query
            query = dict(metadata_key)

            deleted_count = self.collection.delete_many(query).deleted_count

            if deleted_count > 0:
                # print(f"Deleted {deleted_count} documents for Metadata: {metadata_key}")
                pass

        return "Existing chunks successfully deleted from MongoDB."

    def _prepare_new_chunks(self, chunk_group):
        """
        Private method to prepare new embeddings for all chunks of the document,
        generate content hashes for deduplication, and use the hash as document_id.
        """
        new_chunks = []
        last_updated = datetime.now(timezone.utc).isoformat() # Timezone-aware timestamp

        # Track unique content hashes within the same chunk group
        seen_hashes = set()

        for data in chunk_group:
            # Generate a hash of the chunk content, and metadata
            content_hash = self._generate_content_hash(data.content, data.metadata)

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

    def _generate_content_hash(self, content, metadata):
        """
        Private method to generate a SHA256 hash of the combination of content and all metadata elements for deduplication.
        """
        # Combine the content with sorted metadata key-value pairs
        metadata_string = ''.join(f"{key}:{value}" for key, value in sorted(metadata.items()))
        combined_data = f"{content}-{metadata_string}"
        
        # Generate the SHA256 hash
        return hashlib.sha256(combined_data.encode('utf-8')).hexdigest()

    def _add_chunks_to_vector_store(self, new_chunks):
        """
        Private method to add new chunks to the vector store with error handling.
        If a duplicate key error (E11000) occurs, it skips the insertion and logs the event.
        Raises any other exceptions that might occur.
        """
        for chunk in new_chunks:
            try:
                # Attempt to add the document with the custom _id (content_hash)
                self.vector_index.add_documents(documents=[chunk])
            
            except Exception as e:
                # Check if it's a duplicate key error (E11000)
                if 'E11000' in str(e):
                    # print(f"Duplicate document detected for _id {chunk.metadata['_id']}. Skipping insertion.")
                    pass  # Simply pass to continue with the next chunk
                else:
                    # For any other exceptions, raise the error
                    raise e
