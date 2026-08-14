from pydantic import BaseModel, ConfigDict, SecretStr


def mask_key(api_key: str) -> str:
    """The count of hidden characters, then the key's last 5."""
    if len(api_key) <= 5:
        return f"**{len(api_key)}**"
    return f"**{len(api_key) - 5}**{api_key[-5:]}"


class ChudProvider(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    email: str
    name: str
    project_name: str
    project_number: str
    api_key: SecretStr

    @property
    def masked_key(self) -> str:
        # if key is masked, return self (idempotent)
        return mask_key(self.api_key.get_secret_value())

    def __hash__(self) -> int:
        return hash(self.project_number)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ChudProvider)
            and self.project_number == other.project_number
        )

    def __str__(self) -> str:
        return (
            f"ChudProvider[account={self.email!r}, name={self.name!r}, "
            f"project_number={self.project_number!r}, api_key={self.masked_key!r})"
        )
