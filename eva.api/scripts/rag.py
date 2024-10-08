import asyncio
import re
import json
from scripts.models import model_rag
from scripts.rags import BaseRAG, LLMStreamingHandler
from threading import Thread  
from queue import Queue  

class RAG:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data

        self.rag_instance = BaseRAG().get_rag(self.chat_data.rag_type, self.chat_data)
        if self.chat_data.rag_settings.streaming_response:
            self.streamer_queue = Queue()  
            self.result_queue = Queue()  
            self.rag_instance.llm_streaming_callback_handler = LLMStreamingHandler(self.streamer_queue) 

    ##Public Methods

    def get_response(self):
        result = self.rag_instance.get_response()

        if(isinstance(result, model_rag.ChatResponse)):
            response_text = result.response
            source_list = result.sources
        else:
            response_text, extracted_source_ids = self._format_response(result.get("answer",""))
            source_list = self._extract_sources(result.get("source_documents", []), extracted_source_ids)
        
        concept_list = []  
        if self.chat_data.fetch_concepts:
            concept_list = self._extract_concepts(response_text)

        return model_rag.ChatResponse(response=response_text , sources=source_list, concepts=concept_list)
    
    async def get_response_stream(self): 
        try: 
            self._start_generation()
            source_token_buffer = ""  
            inside_source_tag = False
            source_tag_pattern = r"<sources>.*?<\/sources>" 

            while True:
                token = self.streamer_queue.get()  # Get token from the queue
                if token is None:  # End of streaming
                    break

                if token.startswith("<"):
                    inside_source_tag = True  
                    source_token_buffer += token
                    continue

                if inside_source_tag:
                    source_token_buffer += token
                    if source_token_buffer.endswith("</sources>"):  
                        inside_source_tag = False
                        if not re.search(source_tag_pattern, source_token_buffer):
                            yield source_token_buffer
                    continue

                yield token
                self.streamer_queue.task_done()
                await asyncio.sleep(0)  # Ensure smooth streaming

            result = self.result_queue.get()
            response_text, extracted_source_ids = self._format_response(source_token_buffer)
            source_list = self._extract_sources(result.get("source_documents", []), extracted_source_ids)
            concept_list = []
            if self.chat_data.fetch_concepts:
                concept_list = self._extract_concepts(response_text)

            final_response = {
                "sources": [source.dict() for source in source_list],
                "concepts": concept_list
            }

            yield json.dumps(final_response)
            
        except Exception as e:
            yield json.dumps({"error": f"An error occurred during streaming: {str(e)}"})

    ##Private Methods

    def _start_generation(self):
        def run_and_store_result():
            result = self.rag_instance.get_response()
            self.result_queue.put(result)

        thread = Thread(target=run_and_store_result)
        thread.start()

    def _format_response(self, response_text):
        extracted_sources = []
        try:
            source_tag_pattern = r"<sources>(.*?)<\/sources>"
            matches = re.findall(source_tag_pattern, response_text, re.DOTALL)
            for match in matches:
                pans = re.split(r"\s*,\s*|\n+", match)
                pans = [pan for pan in pans if pan.isdigit()]  # Only keep numeric values
                extracted_sources.extend(pans)

            response_text = re.sub(source_tag_pattern, '', response_text).strip()
        except Exception as e:
            print(f"Error while extracting sources: {e}")

        return response_text, extracted_sources

    def _extract_sources(self, sources_documents, extracted_source_ids):
        filtered_sources_list = []
        for doc in sources_documents:
            source_id = doc.metadata.get(next((key for key in doc.metadata if key.lower() == "source"), ""), "")
            if source_id in extracted_source_ids:
                language = doc.metadata.get(next((key for key in doc.metadata if key.lower() == "language"), ""), "")
                if language.lower() == "english" or language.lower() == "en" :
                    filtered_sources_list.append(
                        model_rag.Source(
                            source=source_id,
                            type=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "type"), ""), ""),
                            title=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "title"), ""), ""),
                            country=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "country"), ""), ""),
                            language=language
                        )
                    )
        return filtered_sources_list
        
    def _extract_concepts(self, response_text):
        try:
            from scripts.ner import Concept_NER
            from scripts.models import model_ner

            ner_data = model_ner.NERModelRequest(
                project_directory_name=self.chat_data.prompt_template_directory_name
            )
            concept_ner = Concept_NER(ner_data)

            return concept_ner.fetch_concepts_with_ner_model(response_text)
        
        except Exception as e:
            print(f"Error while extracting concepts: {e}")
            return []
