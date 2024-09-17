from scripts.models import model_rag
from scripts.vectordatabases import BaseDB

import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings

class RAG:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data

    ##Public Methods

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
        
        # Detect the intent first
        intent_name = self._detect_intent()

        # Now invoke the QA model to get the response
        qa = self._get_qa_instance(intent_name)
        result = qa.invoke({"question": self.chat_data.user_input})

        response_text = result.get("answer", "")
        sources_documents = result.get("source_documents", [])
        sources_list = []
        for doc in sources_documents:
            source_value = doc.metadata.get(next((key for key in doc.metadata if key.lower() == "source"), ""), "")
            type_value = doc.metadata.get(next((key for key in doc.metadata if key.lower() == "type"), ""), "")
            if not any(item.source == source_value and item.type == type_value for item in sources_list):
                sources_list.append(
                    model_rag.Source(
                        source=source_value,
                        type=type_value,
                        title=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "title"), ""), ""),
                        country=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "country"), ""), ""),
                        language=doc.metadata.get(next((key for key in doc.metadata if key.lower() == "language"), ""), "")
                    )
                )
        
        return model_rag.ChatResponse(response=response_text, sources=sources_list)

    
    ##Private Methods

    def _load_template(self, project_template_directory_name, template_file_name):
        wd = os.getcwd()

        project_template_directory_path = wd + settings.EVA_SETTINGS_PATH + \
          '/' + project_template_directory_name + \
          '/' + settings.EVA_SETTINGS_ENVIRONMENT_DIRECTORY
          
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

        # Configure retriever with settings from RAGSettings
        qa_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.chat_data.rag_settings.max_chunks_to_retrieve.value,
                "score_threshold": self.chat_data.rag_settings.retrieved_chunks_min_relevance_score.value
            }
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
            model_name=self.chat_data.rag_settings.chat_model_name.value,
            temperature=self.chat_data.rag_settings.temperature.value,
            max_tokens=self.chat_data.rag_settings.max_tokens_for_response.value,
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
        
        history = ChatMessageHistory()
        for conv in self.chat_data.chat_history:
            if conv.role.lower() == 'human':
                history.add_message(HumanMessage(content=conv.message))
            elif conv.role.lower() == 'ai':
                history.add_message(SystemMessage(content=conv.message))
            
        memory = ConversationSummaryMemory.from_messages(
            llm=self.llm_eva,
            chat_memory=history,
            return_messages=True,
            memory_key="history",
            input_key="question"
        )

        summarized_history = memory.buffer
       
        return summarized_history, memory
