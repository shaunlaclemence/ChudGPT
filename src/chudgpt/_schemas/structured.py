from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChudSchema(BaseModel):
    @classmethod
    def pin(cls, **values: Any) -> dict[str, Any]:
        schema = cls.model_json_schema()
        for field, value in values.items():
            if field not in cls.model_fields:
                raise ValueError(f"{cls.__name__} has no field {field!r}")
            # single-value enum, not const: gemini silently ignores const
            schema["properties"][field] = {
                "type": "string",
                "enum": [value.value if isinstance(value, Enum) else value],
            }
            if field not in schema.setdefault("required", []):
                schema["required"].append(field)
        return schema


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    BASH = "bash"
    SQL = "sql"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    SWIFT = "swift"
    C = "C"
    CPP = "C++"
    TXT = "txt"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class GeneratedCode(ChudSchema):
    language: Language = Field(description="Language the code is written in.")
    code: str = Field(
        description="Source only. No markdown fences, no commentary, no prose. "
        "Include every import statement the code needs at the top."
    )
    entrypoint: str = Field(
        description="Name of the function or class to invoke to run this."
    )
    imports: list[str] = Field(
        default_factory=list,
        description="Every module the code imports, standard library included, "
        "as written in the import statement. Example: ['json', 're'].",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Only the imports that need installing before the code runs, "
        "as install names. Standard library modules do not belong here. "
        "Empty if the standard library is enough.",
    )
    explanation: str = Field(
        default="", description="One or two sentences on how it works."
    )


class Classification(ChudSchema):
    label: str = Field(description="The chosen label.")
    confidence: Confidence = Field(description="How sure the model is.")
    reasoning: str = Field(default="", description="Why this label was chosen.")


class SentimentAnalysis(ChudSchema):
    sentiment: Sentiment = Field(description="Overall sentiment of the text.")
    confidence: Confidence = Field(description="How sure the model is.")
    rationale: str = Field(default="", description="Evidence for the judgement.")
