from scripts.models import model_rag
from scripts.vectordatabases import BaseDB

import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from core.config import settings

class RAG:
    def __init__(self, chat_data: model_rag.ChatRequest):
        self.chat_data = chat_data

    ##Public Methods

    def get_response(self):
        intent_name = self._detect_intent()
        qa = self._get_qa_instance(intent_name)
        result = qa.invoke({"question": self.chat_data.user_input})
        
        response_text = result.get("answer", "")
        sources_list = result.get("sources", "")

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
        # Build chat prompt for the detected intent
        dynamic_prompt_content = self._build_chat_prompt(intent_name)

        # Create the prompt template
        prompt_template = PromptTemplate(
            template=dynamic_prompt_content,
            input_variables=['summaries', 'question']
        )

        # Initialize the chat model using settings from RAGSettings
        llm_eva = ChatOpenAI(
            model_name=self.chat_data.rag_settings.chat_model_name.value,
            temperature=self.chat_data.rag_settings.temperature.value,
            max_tokens=self.chat_data.rag_settings.max_tokens_for_response.value,
            openai_api_key=self.chat_data.llm_settings.llm_key,
            streaming=False
        )

        # Get retriever and memory for the conversation
        qa_retriever = self._get_qa_retriever()
        memory = ConversationSummaryBufferMemory(
            memory_key="history",
            input_key="question",
            llm=llm_eva
        )

        # Build the RAG chain
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm_eva,
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt_template,
                "memory": memory
            }
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
