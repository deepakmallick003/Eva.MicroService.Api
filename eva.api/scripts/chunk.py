from scripts.models import model_chunk
from typing import Dict, Any
from langchain.text_splitter import RecursiveJsonSplitter

class ChunkData:
    def __init__(self, chunk_data: model_chunk.ChunkDataRequest):
        self.chunk_data = chunk_data
    
    def process_chunks(self):
        splitter = RecursiveJsonSplitter(max_chunk_size=400, min_chunk_size=200)

        documents = []

        for data_chunk in self.chunk_data.chunks:
            metadata = self.extract_metadata(data_chunk.property_meta_data_map, data_chunk.content)
            if data_chunk.content:
                json_data = data_chunk.content
                split_docs = splitter.create_documents(texts=[json_data], metadatas = [metadata])

                for split_doc in split_docs:
                    document_chunk = self.create_document(
                        text=split_doc.page_content, 
                        metadata=split_doc.metadata  
                    )
                    documents.append(document_chunk)
          
        return model_chunk.ChunkDataResponse(documents=documents)


    def process_chunks_internal(self):

        documents = []

        # Iterate over each chunk (DataChunks)
        for data_chunk in self.chunk_data.chunks:
            # Extract metadata from the property_meta_data_map
            metadata = self.extract_metadata(data_chunk.property_meta_data_map, data_chunk.content)
            
            # Aggregate the content into a single chunk of text
            aggregated_text = ""

            if data_chunk.content:
                for key, value in data_chunk.content.items():
                    # Process simple key-value pairs
                    if isinstance(value, (str, int, float)):
                        aggregated_text += f"{key}: {value}\n"
                    
                    # Process arrays
                    elif isinstance(value, list):
                        array_text = self.process_array(key, value)
                        aggregated_text += f"{array_text}\n"
                    
                    # Process nested objects (dictionaries)
                    elif isinstance(value, dict):
                        nested_text = self.process_nested_dict(key, value)
                        aggregated_text += f"{nested_text}\n"
            
            # Create a single Document with the aggregated text
            document = self.create_document(aggregated_text.strip(), metadata)
            documents.append(document)
            
        # Return the chunked data as a response
        return model_chunk.ChunkDataResponse(documents=documents)
    
    def create_document(self, text: str, metadata: dict) -> model_chunk.DocumentChunk:
        return model_chunk.DocumentChunk(
            content=text,
            metadata=metadata
        )

    def process_array(self, key: str, array: list) -> str:
        """
        Process arrays and format them into a string with indentation.
        """
        result = f"{key}:\n"
        for item in array:
            result += f"\t- {item}\n"  # Indent list items
        return result.strip()  # Remove trailing newline
    
    def process_nested_dict(self, parent_key: str, nested_dict: dict) -> str:
        """
        Process nested dictionaries and format them into a structured text.
        """
        result = f"{parent_key}:\n"
        for key, value in nested_dict.items():
            result += f"\t{key}: {value}\n"  # Indent nested key-value pairs
        return result.strip()  # Remove trailing newline
    
    def extract_metadata(self, meta_data_map: Dict[str, str], content: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract metadata based on the property_meta_data_map.
        """
        metadata = {}
        if meta_data_map:
            for meta_key, content_key in meta_data_map.items():
                if content_key in content:
                    metadata[meta_key] = str(content[content_key])  # Convert to string for consistency
        return metadata
