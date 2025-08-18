from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    feature_genai: bool = Field(default=False)
    mock_llm: bool = Field(default=True)
    ocr_enabled: bool = Field(default=False)
    pii_redact: bool = Field(default=False)
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_settings() -> GenAISettings:
    return GenAISettings()
