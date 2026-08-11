from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChudProvider(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    project_name: str
    project_number: str
    api_key: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChudProvider) and self.id == other.id


class ChudQuota(BaseModel):
    slug: str
    rpd: int
    rpm: int
    tpm: int
    inputs: list[str]


class ChudUsageRecord(BaseModel):
    model: str
    provider: ChudProvider
    created_at: datetime
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int


class ChudUsageSummary(BaseModel):
    usage: list[ChudUsageRecord]
    quotas: list[ChudQuota]
