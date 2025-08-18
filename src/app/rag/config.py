from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    feature_genai: bool = Field(default=False)
    mock_llm: bool = Field(default=True)
    ocr_enabled: bool = Field(default=False)
    pii_redact: bool = Field(default=False)
    """
    Embeddings - just use all-MiniLM-L6-v2 for now; 384 is the default dims for this model

    Proof:
    from sentence_transformers import SentenceTransformer
    dim = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").get_sentence_embedding_dimension()
    print(dim) # 384
    """
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embed_backend: str = Field(default="hash")  # 'hash' or 'st'/SentenceTransformer
    embed_dim: int = Field(default=384)
    index_backend: str = Field(default="np")  # 'np' or 'faiss' when it is supported


@lru_cache(maxsize=1)
def get_settings() -> GenAISettings:
    return GenAISettings()
