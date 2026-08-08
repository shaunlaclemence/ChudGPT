from pydantic import BaseModel


class Quota(BaseModel):
    slug: str
    rpd: int
    rpm: int
    tpm: int
    inputs: list[str]
