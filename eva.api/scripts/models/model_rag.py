from .model_base import *
from .model_ner import Concept

class Source(BaseModel):
    source: str
    type: str
    title: str
    country: Optional[str]
    language: Optional[str]

class IntentDetails(BaseModel):
    filename: str
    description: Optional[str] = None
    required_fields_prior_responding: Optional[List[str]] = None

class ConversationMessage(BaseModel):
    role: RAGRoles 
    message: str

class ChatRequest(BaseModel):
    db_type: VectorDatabaseType 
    db_settings: DBSettings 
    llm_settings: LLMSettings
    rag_type: RAGStrategy = RAGStrategy.Version3
    rag_settings: RAGSettings = RAGSettings()    
    strict_follow_up: bool = False
    prompt_template_directory_name: str
    base_prompt_template_file_name: Optional[str] = "base_template.txt"
    intent_detection_prompt_template_file_name: Optional[str] = "detect_intent.txt"
    intent_details: Optional[Dict[str, IntentDetails]] = None
    memory_prompt_template_file_name: Optional[str] = "memory_summarizer.txt"
    follow_up_prompt_template_file_name: Optional[str] = "follow_up.txt"
    free_flowing_prompt_template_file_name: Optional[str] = "free_flowing.txt"
    fetch_concepts: bool = False
    chat_history: Optional[List[ConversationMessage]] = None
    user_input: str

class ChatResponse(BaseModel):
    response: str  
    sources: Optional[List[Source]] = None
    concepts: Optional[List[Concept]] = None

