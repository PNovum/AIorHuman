from pydantic import BaseModel, StrictStr

class TextRequest(BaseModel):
    text: StrictStr
