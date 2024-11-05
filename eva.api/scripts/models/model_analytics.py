from .model_base import *

class SourceAppAnalyticsRequest(BaseModel):
    source_project_name: str = Field(
        ..., 
        description="A Short name of the Project from where you are calling the EVA Analytics endpoint"
    )
    source_app_data: Dict[str, Any]

class AnalyticsRequest(BaseModel):
    chatbot_data: Dict[str, Any]