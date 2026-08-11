from pydantic import BaseModel


class ChudQuota(BaseModel):
    slug: str
    rpd: int
    rpm: int
    tpm: int
    inputs: list[str]
