from .model_base import *

class IntentDetails(BaseModel):
    filename: str
    description: Optional[str] = None

class ConversationMessage(BaseModel):
    role: RAGRoles 
    message: str

class ChatRequest(BaseModel):
    db_type: VectorDatabaseType 
    db_settings: DBSettings 
    llm_settings: LLMSettings
    rag_settings: RAGSettings = RAGSettings()
    user_input: str
    chat_history: Optional[List[ConversationMessage]] = None
    prompt_template_directory_name: str
    base_prompt_template_file_name: str
    intent_detection_prompt_template_file_name: Optional[str] = None
    intent_details: Optional[Dict[str, IntentDetails]] = None

class ChatResponse(BaseModel):
    response: str  
    sources: Optional[str] = None

