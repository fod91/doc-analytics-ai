from pydantic import BaseModel


class IngestItem(BaseModel):
    source: str
    text: str
    label: str | None = None
