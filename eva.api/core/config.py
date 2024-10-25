# Application Settings
# From the enviornments vaiables
from pydantic import Field
from pydantic_settings import BaseSettings

class ApplicationSettings(BaseSettings):
    # DEPLOYED_BASE_PATH: str = Field(default='/eva-api',env='DEPLOYED_BASE_PATH')
    # DEPLOYED_BASE_PATH: str = Field(default='/',env='DEPLOYED_BASE_PATH')
    # EVA_API_ClientId: str = Field(default='',env='EVA__API__ClientId')
    # CABI_TenantId: str = Field(default='', env='CABI__TenantId')

    DEPLOYED_BASE_PATH: str = Field(default="/",alias='DEPLOYED_BASE_PATH')
    EVA_API_ClientId: str = Field(default="",alias='AzureAd__ClientId')
    CABI_TenantId: str = Field(default="", alias='AzureAd__TenantId')
    EVA_SETTINGS_ENVIRONMENT_DIRECTORY: str = Field(default='', alias='FileStoreSettings__StorageDirectory')


class Settings(ApplicationSettings):
    PROJECT_NAME: str = 'EVA API'
    DOC_URL: str = '/swagger'
    EVA_SETTINGS_PATH: str = '/evasettings'

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()