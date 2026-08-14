from pydantic import BaseModel


class Plugin(BaseModel):
    module: str
    service: str
    extra: str
