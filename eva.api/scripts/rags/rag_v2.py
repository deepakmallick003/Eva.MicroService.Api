from scripts.models import model_rag
from scripts.vectordatabases import BaseDB

import os
import re
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings

class RAG_V2:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data
        self.llm_streaming_callback_handler = None

    ##Public Methods

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
        
        intent_name = self._detect_intent()

        if self.chat_data.rag_settings.streaming_response and self.llm_streaming_callback_handler:
            llm_params["streaming"] = True
            llm_params["callback_manager"] = [self.llm_streaming_callback_handler]
        self.llm_eva = ChatOpenAI(**llm_params)

        qa = self._get_qa_instance(intent_name)
        
        result = qa.invoke({"question": self.chat_data.user_input})
        return result
    
    ##Private Methods

    def _load_template(self, project_template_directory_name, template_file_name):
        wd = os.getcwd()

        project_template_directory_path = wd + settings.EVA_SETTINGS_PATH + \
          '/' + project_template_directory_name + \
          '/' + settings.EVA_SETTINGS_ENVIRONMENT_DIRECTORY + \
          '/' + self.chat_data.rag_type
          
        template_file_path = project_template_directory_path+ '/' + template_file_name

        with open(template_file_path, "r") as file:
            return file.read()

    def _build_intent_detection_prompt(self):
        # Load intent detection template
        intent_detection_template = self._load_template(
            self.chat_data.prompt_template_directory_name, 
            self.chat_data.intent_detection_prompt_template_file_name
        )
        
        # Dynamically build the intent list from intent details
        intent_list = "\n".join(
            [f'- "{intent_name}": {intent_data.description}' for intent_name, intent_data in self.chat_data.intent_details.items()]
        )
        
        # Build the full prompt for intent detection
        prompt = intent_detection_template.format(
            user_input=self.chat_data.user_input,
            history=self.chat_data.chat_history,
            intent_list=intent_list
        )
        return prompt

    def _build_chat_prompt(self, intent_name):
        # Load base template
        base_template = self._load_template(self.chat_data.prompt_template_directory_name, self.chat_data.base_prompt_template_file_name)
        
        # Load intent-specific template or default to generic message
        if intent_name == "none":
            intent_template = ""
        else:
            intent_filename = self.chat_data.intent_details.get(intent_name).filename
            intent_template = self._load_template(self.chat_data.prompt_template_directory_name, intent_filename)
        
        # Build the full prompt using the base and intent templates
        return base_template.format(
            subinstructions=intent_template,
            history=self.chat_data.chat_history,
            summaries="{summaries}",
            question="{question}"
        )

    def _get_qa_retriever(self):
        # Initialize embeddings with settings from LLMSettings
        llm_embeddings = OpenAIEmbeddings(
            model=self.chat_data.llm_settings.embedding_model_name,
            openai_api_key=self.chat_data.llm_settings.llm_key
        )

        # Get the vector database instance
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

        # Call the retriever
        qa_retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        return qa_retriever

    def _get_qa_instance(self, intent_name):
        dynamic_prompt_content = self._build_chat_prompt(intent_name)
            
        prompt_template = PromptTemplate(
            template=dynamic_prompt_content,
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

    def _detect_intent(self):
        # Build the intent detection prompt
        intent_prompt = self._build_intent_detection_prompt()

        # Initialize the LLM model for intent detection
        llm_eva = ChatOpenAI(
            model_name=self.chat_data.rag_settings.chat_model_name,
            temperature=self.chat_data.rag_settings.temperature,
            max_tokens=self.chat_data.rag_settings.max_tokens_for_response,
            openai_api_key=self.chat_data.llm_settings.llm_key,
            streaming=False
        )

        # Create the intent detection prompt template
        prompt_template = PromptTemplate(
            template=intent_prompt,
            input_variables=["user_input", "history", "intent_list"]
        )

        # Detect the intent by invoking the chain
        intent_chain = RunnableSequence(prompt_template, llm_eva)
        intent_result = intent_chain.invoke({
            "user_input": self.chat_data.user_input,
            "history": self.chat_data.chat_history,
            "intent_list": "\n".join([f'- "{intent_name}"' for intent_name in self.chat_data.intent_details.keys()])
        })

        # Access content from AIMessage and handle potential "none" category
        detected_intent = intent_result.content.strip()
        if detected_intent not in self.chat_data.intent_details:
            return "none"
        return detected_intent
    
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
