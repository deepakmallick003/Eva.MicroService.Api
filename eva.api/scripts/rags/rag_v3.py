from scripts.models import model_rag
from scripts.vectordatabases import BaseDB

import os
import re
import json
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings

class RAG_V3:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data

    ## Public Methods
    def get_response(self):
        self.llm_eva = ChatOpenAI(
            model_name=self.chat_data.rag_settings.chat_model_name,
            temperature=self.chat_data.rag_settings.temperature,
            max_tokens=self.chat_data.rag_settings.max_tokens_for_response,
            openai_api_key=self.chat_data.llm_settings.llm_key
        )

        self.summarized_history = ""
        self.memory = None
        if self.chat_data.chat_history:
            self.summarized_history, self.memory = self._summarize_history()

        base_response = self._run_base_prompt()
        if base_response is not None:
            return model_rag.ChatResponse(response=base_response, sources=[])

        self._detect_intent_and_rephrase_query()

        if self.chat_data.strict_follow_up:
            follow_up_question = self._follow_up_questions()
            if follow_up_question:
                return model_rag.ChatResponse(response=follow_up_question, sources=[])

        qa = self._get_qa_instance()
        result = qa.invoke({"question": self.chat_data.user_input})
        
        response_text, extracted_source_ids = self._format_response(result.get("answer", ""))
        source_list = self._extract_sources(result.get("source_documents", []), extracted_source_ids)
        return model_rag.ChatResponse(response=response_text, sources=source_list)

        
    ## Private Methods

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
        follow_up_result = follow_up_chain.invoke({
            "missing_fields": ", ".join(missing_fields),
            "intent_name": self.detected_intent
        })
    
        return follow_up_result.content.strip()

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
                if language.lower() == "english":
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

    def _get_qa_instance(self):
        chat_prompt_content = self._build_chat_prompt()
            
        prompt_template = PromptTemplate(
            template=chat_prompt_content,
            input_variables=['summaries', 'question']
        )
    
        qa_retriever = self._get_qa_retriever()

        if self.memory:
            chain_type_kwargs = {
                "verbose": False,
                "prompt": prompt_template,
                "memory": self.memory  # Include memory if available
            }
        else:
            chain_type_kwargs = {
                "verbose": False,
                "prompt": prompt_template
            }

        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm_eva,
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
    
        return qa

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
        intent_result = intent_chain.invoke({
            "user_input": self.chat_data.user_input,  
            "history": self.summarized_history,  
            "intent_list": "\n".join([f'- "{intent_name}"' for intent_name in self.chat_data.intent_details.keys()]),
            "merged_key_fields": ", ".join(merged_key_fields)
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
        base_result = base_chain.invoke({
            "user_input": self.chat_data.user_input,  
            "history": self.summarized_history
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