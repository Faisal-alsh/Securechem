"""
Configuration settings for the backend
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Model Settings
    BASE_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    USE_8BIT_QUANTIZATION: bool = False  # Set to True if you have GPU
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # RAG Settings
    RAG_TOP_K: int = 3
    MAX_CONTEXT_LENGTH: int = 1000

    # Security Settings
    CHEM_PASSWORD: str = "1122"
    BIO_PASSWORD: str = "3344"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
