from pydantic import BaseModel


class IngestItem(BaseModel):
    source: str
    text: str
    label: str | None = None


class SentimentSummary(BaseModel):
    n: int
    pos: int
    neg: int
    neu: int
