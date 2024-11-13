from scripts.models import model_rag
from scripts.vectordatabases import BaseDB

import os
import re
import json
import time
import tiktoken
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings
from collections import defaultdict

class RAG_V4:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data
        self.llm_streaming_callback_handler = None
        self.eva_analytics = []

        self.max_chunks_per_search = 20
        self.top_k_chunks_for_summary = 15
        self.weight_chunk_relevancy = 0.6  # Increase impact of chunk relevancy
        self.weight_frequency = 0.4  # Increase impact of frequency

    ## Public Methods
    def get_response(self):
        llm_params = {
            "openai_api_key": self.chat_data.llm_settings.llm_key,
            "max_tokens": self.chat_data.rag_settings.max_tokens_for_response,
            "model_name": self.chat_data.rag_settings.chat_model_name,
            "temperature": self.chat_data.rag_settings.temperature,
            "frequency_penalty": self.chat_data.rag_settings.frequency_penalty,
            "presence_penalty": self.chat_data.rag_settings.presence_penalty
        }

        self.llm_eva = ChatOpenAI(**llm_params)

        self.summarized_history = ""
        self.memory = None
        if self.chat_data.chat_history:
            self.summarized_history, self.memory = self._summarize_history()

        base_response = self._run_base_prompt()
        if base_response is not None:
            return model_rag.ChatResponse(response=base_response, sources=[]), self.eva_analytics

        self._detect_intent_and_rephrase_query()

        if self.chat_data.strict_follow_up:
            follow_up_question = self._follow_up_questions()
            if follow_up_question:
                return model_rag.ChatResponse(response=follow_up_question, sources=[]), self.eva_analytics

        if self.chat_data.rag_settings.streaming_response and self.llm_streaming_callback_handler:
            llm_params["streaming"] = True
            llm_params["callback_manager"] = [self.llm_streaming_callback_handler]
            self.llm_eva = ChatOpenAI(**llm_params)

        result = self._invoke_qa()
        return result, self.eva_analytics

    ## Private Methods

    def _invoke_qa(self):
        source_documents, chunk_summary = self._vector_search()
        
        self.chat_prompt_content = self._build_chat_prompt()
        prompt_template = PromptTemplate(
            template=self.chat_prompt_content,
            input_variables=['summaries', 'question']
        )

        qa_chain = RunnableSequence(prompt_template, self.llm_eva)
        qa_input = {
            "summaries": chunk_summary,
            "question": self.chat_data.user_input
        }
        
        start_time = time.time()
        qa_result = qa_chain.invoke(qa_input)
        latency = (time.time()) - start_time
    
        # Extract the answer
        output_text = qa_result.content
        
        ## Append analytics data ##
        # Calculate token usage for analytics
        try:
            tokenizer = tiktoken.encoding_for_model(self.chat_data.rag_settings.chat_model_name)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
        formatted_prompt = self.chat_prompt_content.format(
            summaries=chunk_summary,
            question=self.chat_data.user_input
        )
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        completion_tokens = len(tokenizer.encode(output_text))
        total_tokens = prompt_tokens + completion_tokens
        self.eva_analytics.append({
            "event": "invoke_qa",
            "response_time_in_seconds": latency,
            "response_metadata": {
                "model_name": self.chat_data.rag_settings.chat_model_name,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
        })
    
        return {
            "answer": output_text,
            "source_list": source_documents
        }

    def _vector_search(self):
        self._get_vector_store()
        self._paraphrase_query()
        
        source_documents = []
        unranked_documents = []
        stats_dict = defaultdict(lambda: {
            "total_relevancy": 0,
            "chunk_count": 0,
            "chunks": {}
        })

        for phrase in self.paraphrases:
            results = self.vector_store.similarity_search_with_relevance_scores(query=phrase, k=self.max_chunks_per_search)
            
            for doc, score in results:
                doc_id = doc.metadata.get("_id")
                source = doc.metadata.get("source", "unknown")
    
                # Update source-level stats (without average source relevancy)
                stats_dict[source]["total_relevancy"] += score
                stats_dict[source]["chunk_count"] += 1
    
                # Update individual chunk stats within each source
                if doc_id not in stats_dict[source]["chunks"]:
                    stats_dict[source]["chunks"][doc_id] = {"total_relevancy": 0, "count": 0}
                stats_dict[source]["chunks"][doc_id]["total_relevancy"] += score
                stats_dict[source]["chunks"][doc_id]["count"] += 1
    
                # Add unique documents to unranked_documents list
                if doc_id not in {d['doc_id'] for d in unranked_documents}:
                    clean_metadata = {
                        key: str(value) if not isinstance(value, (str, dict)) else value
                        for key, value in doc.metadata.items() if key not in {"_id", "embedding"}
                    }
                    
                    unranked_documents.append(                        
                    {
                        "content": doc.page_content,
                        "metadata": clean_metadata,
                        "relevancy_score": score,
                        "doc_id": doc_id
                    })
    
        # Calculate average relevancies and prepare stats_dict
        weighted_chunks = []
        for source, source_data in stats_dict.items():
            for doc_id, chunk_data in source_data["chunks"].items():
                # Calculate average chunk relevancy
                avg_chunk_relevancy = chunk_data["total_relevancy"] / chunk_data["count"]
                chunk_frequency = chunk_data["count"]
                
                # Calculate the weighted score for this chunk
                weighted_score = (
                    self.weight_chunk_relevancy * avg_chunk_relevancy +
                    self.weight_frequency * (chunk_frequency ** 0.5)  # Apply square root to balance frequency's impact
                )
    
                # Append chunk info to weighted_chunks list
                weighted_chunks.append({
                    "source": source,
                    "doc_id": doc_id,
                    "average_chunk_relevancy": avg_chunk_relevancy,
                    "chunk_frequency": chunk_frequency,
                    "weighted_score": weighted_score
                })
    
        # Filter based on weighted score threshold
        filtered_chunks = sorted(weighted_chunks, key=lambda x: x["weighted_score"], reverse=True)
        filtered_chunks = filtered_chunks[:self.top_k_chunks_for_summary]
    
        selected_doc_ids = {chunk["doc_id"] for chunk in filtered_chunks}
        ranked_and_filtered_chunks = [doc for doc in unranked_documents if doc["doc_id"] in selected_doc_ids]
    
        added_sources = set()
        for doc in ranked_and_filtered_chunks:
            meta = doc["metadata"]
            source_value = meta.get(next((key for key in meta if key.lower() == "source"), ""), "")
            if source_value not in added_sources:
                source_documents.append(
                    model_rag.Source(
                        source=source_value,
                        type=meta.get(next((key for key in meta if key.lower() == "type"), ""), ""),
                        title=meta.get(next((key for key in meta if key.lower() == "title"), ""), ""),
                        country=meta.get(next((key for key in meta if key.lower() == "country"), ""), ""),
                        language=meta.get(next((key for key in meta if key.lower() == "language"), ""), ""),
                    )
                )
                added_sources.add(source_value)

        merged_ranked_filtered_chunks = self._extract_content_by_source(ranked_and_filtered_chunks)
        chunk_summary = self._chunks_summarizer(merged_ranked_filtered_chunks)

        return source_documents, chunk_summary
    
    def _extract_content_by_source(self, unranked_documents):
        source_chunks = {}
        for doc in unranked_documents:
            source = doc["metadata"].get("source", "unknown")
            content = doc.get("content", "")
            chunk_sequence = int(doc["metadata"].get("chunk", 0))  # Use "chunk" as an integer sequence
            if "text:" in content:
                text_chunk = content.split("text:", 1)[1].strip()
            else:
                text_chunk = content.strip()
    
            if source not in source_chunks:
                source_chunks[source] = []
            source_chunks[source].append({"content": text_chunk, "chunk_sequence": chunk_sequence})
    
        content_by_source = []
        
        for source, chunks in source_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda x: x["chunk_sequence"])
            merged_content = " ".join(chunk["content"] for chunk in sorted_chunks)
            content_by_source.append({
                "source": source,
                "content": merged_content
            })
    
        return content_by_source

    def _chunks_summarizer(self, merged_chunks_by_source):
        text_chunks_string = "\n".join(str(chunk) for chunk in merged_chunks_by_source)
        chunk_summary = text_chunks_string       
        
        try:
            chunks_summarizer_prompt = self._load_template(
                self.chat_data.prompt_template_directory_name, 
                self.chat_data.chunk_summarizer_prompt_template_file_name
            )
    
            prompt_template = PromptTemplate(
                template=chunks_summarizer_prompt,
                input_variables=["user_input", "text_chunks"]
            )
            
            chunks_summarizer_chain = RunnableSequence(prompt_template, self.llm_eva)
            input = {
                "user_input": self.chat_data.user_input,
                "text_chunks": text_chunks_string,
            }

            start_time = time.time()
            chunks_summarizer_result = chunks_summarizer_chain.invoke(input)
            latency = (time.time()) - start_time
            

            self.eva_analytics.append({
                "event": "_chunks_summarizer",
                "response_time_in_seconds": latency,
                "response_metadata": chunks_summarizer_result.response_metadata
            })
                
            if chunks_summarizer_result.content:
                chunk_summary = chunks_summarizer_result.content
        except (json.JSONDecodeError, AttributeError, KeyError) as e:        
            chunk_summary = text_chunks_string

        return chunk_summary
        
    def _paraphrase_query(self):
        self.paraphrases = [self.chat_data.user_input]
    
        paraphrase_prompt = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.paraphrasing_prompt_template_file_name
        )

        prompt_template = PromptTemplate(
            template=paraphrase_prompt,
            input_variables=["user_input"]
        )
        
        paraphrase_chain = RunnableSequence(prompt_template, self.llm_eva)
        input = {
            "user_input": self.chat_data.user_input
        }

        start_time = time.time()
        paraphrase_result = paraphrase_chain.invoke(input)
        latency = (time.time()) - start_time

        self.eva_analytics.append({
            "event": "_paraphrase_query",
            "response_time_in_seconds": latency,
            "response_metadata": paraphrase_result.response_metadata
        })
        
        try:
            cleaned_content = re.sub(r"```json|```", "", paraphrase_result.content).strip()
            paraphrase_result_json = json.loads(cleaned_content)
            if "paraphrases" in paraphrase_result_json:
                self.paraphrases.extend(paraphrase_result_json["paraphrases"])

        except (json.JSONDecodeError, AttributeError, KeyError) as e:        
            pass     

    def _get_vector_store(self):
        self.llm_eva = ChatOpenAI(
            model_name=self.chat_data.rag_settings.chat_model_name,
            temperature=self.chat_data.rag_settings.temperature,
            max_tokens=self.chat_data.rag_settings.max_tokens_for_response,
            openai_api_key=self.chat_data.llm_settings.llm_key
        )

        llm_embeddings = OpenAIEmbeddings(
            model=self.chat_data.llm_settings.embedding_model_name,
            openai_api_key=self.chat_data.llm_settings.llm_key
        )
        
        db_instance = BaseDB().get_vector_db(
            self.chat_data.db_type,
            self.chat_data.db_settings,
            llm_embeddings
        )
        
        self.vector_store = db_instance.vector_index


    def _follow_up_questions(self):
        follow_up_prompt = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.follow_up_prompt_template_file_name
        )
        
        prompt_template = PromptTemplate(
            template=follow_up_prompt,
            input_variables=["missing_fields", "intent_name"]
        )
    
        # Collect missing fields
        missing_fields = []
        
        if self.detected_intent in self.chat_data.intent_details:
            intent_detail_obj = self.chat_data.intent_details[self.detected_intent]
            required_fields = intent_detail_obj.required_fields_prior_responding or []
            for field in required_fields:
                if field not in self.extracted_fields:
                    missing_fields.append(field)
    
        if not missing_fields:
            return None 
    
        follow_up_chain = RunnableSequence(prompt_template, self.llm_eva)
        input = {
            "missing_fields": ", ".join(missing_fields),
            "intent_name": self.detected_intent
        }

        start_time = time.time()
        follow_up_result = follow_up_chain.invoke(input)
        latency = (time.time()) - start_time

        self.eva_analytics.append({
            "event": "_follow_up_questions",
            "response_time_in_seconds": latency,
            "response_metadata": follow_up_result.response_metadata
        })
    
        return follow_up_result.content.strip()

    def _load_template(self, project_template_directory_name, template_file_name):
        wd = os.getcwd()

        project_template_directory_path = wd + settings.EVA_SETTINGS_PATH + \
          '/' + project_template_directory_name + \
          '/' + settings.EVA_SETTINGS_ENVIRONMENT_DIRECTORY + \
          '/' + self.chat_data.rag_type
          
        template_file_path = project_template_directory_path+ '/' + template_file_name

        with open(template_file_path, "r") as file:
            return file.read()

    def _build_chat_prompt(self):
        if self.detected_intent == "other":
            chat_prompt_filename = self.chat_data.free_flowing_prompt_template_file_name
            chat_template = self._load_template(self.chat_data.prompt_template_directory_name, chat_prompt_filename)           
            return chat_template.format(
                history=self.summarized_history,
                summaries="{summaries}",
                question="{question}"
            )
        else:                        
            chat_prompt_filename = self.chat_data.intent_details.get(self.detected_intent).filename
            chat_template = self._load_template(self.chat_data.prompt_template_directory_name, chat_prompt_filename)
            key_fields = self.chat_data.intent_details[self.detected_intent].required_fields_prior_responding or ""
            
            return chat_template.format(
                history=self.summarized_history,
                key_fields=key_fields,
                summaries="{summaries}",
                question="{question}"
            )

    def _get_qa_retriever(self):
        llm_embeddings = OpenAIEmbeddings(
            model=self.chat_data.llm_settings.embedding_model_name,
            openai_api_key=self.chat_data.llm_settings.llm_key
        )
    
        db_instance = BaseDB().get_vector_db(
            self.chat_data.db_type,
            self.chat_data.db_settings,
            llm_embeddings
        )
        vector_store = db_instance.vector_index
        
        search_type = self.chat_data.rag_settings.retriever_search_settings.search_type
        search_kwargs = {
            "k": self.chat_data.rag_settings.retriever_search_settings.max_chunks_to_retrieve
        }
        if search_type == model_rag.RetrieverSearchType.Similarity_Score_Threshold:
            search_kwargs["score_threshold"] = self.chat_data.rag_settings.retriever_search_settings.retrieved_chunks_min_relevance_score
        elif search_type == model_rag.RetrieverSearchType.MMR:
            search_kwargs["fetch_k"] = self.chat_data.rag_settings.retriever_search_settings.fetch_k
            search_kwargs["lambda_mult"] = self.chat_data.rag_settings.retriever_search_settings.lambda_mult

        qa_retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return qa_retriever

    def _get_key_fields_for_lookup(self):
        merged_key_fields = set()
        for _, details in self.chat_data.intent_details.items():
            required_fields = details.required_fields_prior_responding or []
            merged_key_fields.update(required_fields)
    
        return list(merged_key_fields)
    
    def _detect_intent_and_rephrase_query(self):
        self.detected_intent = "other"
        self.extracted_fields = {}
        merged_key_fields = self._get_key_fields_for_lookup()
    
        intent_prompt = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.intent_detection_prompt_template_file_name
        )

        prompt_template = PromptTemplate(
            template=intent_prompt,
            input_variables=["user_input", "history", "intent_list", "merged_key_fields"]
        )
        
        intent_chain = RunnableSequence(prompt_template, self.llm_eva)
        input = {
            "user_input": self.chat_data.user_input,  
            "history": self.summarized_history,  
            "intent_list": "\n".join([f'- "{intent_name}": {intent_detail.description}' 
                              for intent_name, intent_detail in self.chat_data.intent_details.items() if intent_detail.description]),
            "merged_key_fields": ", ".join(merged_key_fields)
        }

        start_time = time.time()
        intent_result = intent_chain.invoke(input)
        latency = (time.time()) - start_time

        self.eva_analytics.append({
            "event": "_detect_intent_and_rephrase_query",
            "response_time_in_seconds": latency,
            "response_metadata": intent_result.response_metadata
        })
        
        try:
            cleaned_content = re.sub(r"```json|```", "", intent_result.content).strip()
            intent_result_json = json.loads(cleaned_content)
            
            if all(key in intent_result_json for key in ["detected_intent", "rephrased_query", "extracted_fields"]):
                self.detected_intent = intent_result_json.get("detected_intent", self.detected_intent).strip().lower()
                self.chat_data.user_input = intent_result_json.get("rephrased_query", self.chat_data.user_input).strip()
                self.extracted_fields = intent_result_json.get("extracted_fields", self.extracted_fields)

        except (json.JSONDecodeError, AttributeError, KeyError) as e:        
            pass        
                
    def _run_base_prompt(self):
        base_prompt = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.base_prompt_template_file_name
        )
        
        prompt_template = PromptTemplate(
            template=base_prompt,
            input_variables=["user_input", 'history']
        )
        
        base_chain = RunnableSequence(prompt_template, self.llm_eva)
        input = {
            "user_input": self.chat_data.user_input,  
            "history": self.summarized_history
        }
       
        start_time = time.time()
        base_result = base_chain.invoke(input)
        latency = (time.time()) - start_time
        
        self.eva_analytics.append({
            "event": "_run_base_prompt",
            "response_time_in_seconds": latency,
            "response_metadata": base_result.response_metadata
        })

        base_response = base_result.content.strip().strip(' "\'')
        if base_response.lower() == "none":
            return None
        return base_response
    
    def _summarize_history(self): 
        if not self.chat_data.chat_history:
            return "", None

        def trim_message(message, max_lines=2):
            lines = message.splitlines()
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + "..."
            return message
        
        history = ChatMessageHistory()
        for conv in self.chat_data.chat_history:
            if conv.role.lower() == 'human':
                history.add_message(HumanMessage(content=conv.message))
            elif conv.role.lower() == 'ai':
                trimmed_message = trim_message(conv.message)
                history.add_message(SystemMessage(content=trimmed_message))

        memory_template = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.memory_prompt_template_file_name
        )
        
        custom_prompt = PromptTemplate(
            input_variables=['new_lines', 'summary'],
            template=memory_template
        )
        
        memory = ConversationSummaryBufferMemory(
            llm=self.llm_eva,
            max_token_limit=self.chat_data.rag_settings.max_tokens_for_history,
            prompt=custom_prompt,
            chat_memory=history,
            return_messages=True,
            memory_key="history",
            input_key="question"
        )

        memory.prune()
        summarized_history = memory.predict_new_summary(memory.chat_memory.messages, "")
       
        return summarized_history, memory
    