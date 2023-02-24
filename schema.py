from pydantic import BaseModel


class WordSchema(BaseModel):
    word_text: str
