from pydantic import BaseModel, Field


class Request(BaseModel):
    # Request model
    image: str = Field()

class Response(BaseModel):
    # Response model
    prediction: int = Field(...)
