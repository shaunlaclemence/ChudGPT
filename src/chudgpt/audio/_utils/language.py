from __future__ import annotations

import re
from typing import ClassVar


class LanguageRules:
    """Spoken language as one ISO 639-1 code, whatever the model called it.

    Models answer "en", "eng" and "English" interchangeably, sometimes within
    one file, which turns a single language into three entries in any union
    taken over them.
    """

    UNKNOWN = "UND"
    SPLIT = re.compile(r"[-_\s]")

    CODES: ClassVar[dict[str, tuple[str, ...]]] = {
        "EN": ("english", "eng"),
        "FR": ("french", "fra", "fre", "francais", "français"),
        "ES": ("spanish", "spa", "castellano", "espanol", "español"),
        "DE": ("german", "deu", "ger", "deutsch"),
        "IT": ("italian", "ita", "italiano"),
        "PT": ("portuguese", "por", "portugues", "português"),
        "NL": ("dutch", "nld", "dut", "nederlands"),
        "RU": ("russian", "rus"),
        "ZH": ("chinese", "zho", "chi", "mandarin", "cmn", "putonghua", "pinyin"),
        "JA": ("japanese", "jpn", "nihongo"),
        "KO": ("korean", "kor"),
        "AR": ("arabic", "ara"),
        "HI": ("hindi", "hin"),
        "EL": ("greek", "ell", "gre"),
        "LA": ("latin", "lat"),
        "SV": ("swedish", "swe"),
        "NO": ("norwegian", "nor"),
        "DA": ("danish", "dan"),
        "FI": ("finnish", "fin"),
        "PL": ("polish", "pol"),
        "TR": ("turkish", "tur"),
        "HE": ("hebrew", "heb"),
        "VI": ("vietnamese", "vie"),
        "TH": ("thai", "tha"),
        "ID": ("indonesian", "ind"),
        "MS": ("malay", "msa", "may"),
        "TL": ("tagalog", "tgl", "filipino"),
        "UK": ("ukrainian", "ukr"),
        "CS": ("czech", "ces", "cze"),
        "RO": ("romanian", "ron", "rum"),
        "HU": ("hungarian", "hun"),
    }

    ALIASES: ClassVar[dict[str, str]] = {
        alias: code for code, names in CODES.items() for alias in (*names, code.lower())
    }

    @classmethod
    def code(cls, value: str | None) -> str:
        if not value or not value.strip():
            return cls.UNKNOWN
        cleaned = value.split("(")[0].strip().lower()
        if cleaned in cls.ALIASES:
            return cls.ALIASES[cleaned]
        # "en-US", "chinese pinyin" and "French Canadian" all key off the head
        head = cls.SPLIT.split(cleaned)[0]
        return cls.ALIASES.get(head, head.upper())

    @classmethod
    def same(cls, left: str | None, right: str | None) -> bool:
        return cls.code(left) == cls.code(right)

    @classmethod
    def union(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(cls.code(v) for v in values if v))
