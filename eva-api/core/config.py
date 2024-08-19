# Application Settings
# From the enviornments vaiables
from pydantic import Field
from pydantic_settings import BaseSettings

class ApplicationSettings(BaseSettings):
    DEPLOYED_BASE_PATH: str = Field(default='/eva-api',env='DEPLOYED_BASE_PATH')


class Settings(ApplicationSettings):
    PROJECT_NAME: str = 'EVA API'
    DOC_URL: str = '/swagger'

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()