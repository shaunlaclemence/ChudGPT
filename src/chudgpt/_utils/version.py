from importlib.metadata import PackageNotFoundError, version

PACKAGE = "chudgpt"
REPO = "https://github.com/shaunlaclemence/ChudGPT.git"


class VersionRules:
    # pyproject.toml is the only place the number lives; this reads it back out
    # of the installed distribution metadata so nothing else hardcodes it
    FALLBACK = "0.0.0+unknown"

    @classmethod
    def installed(cls) -> str:
        try:
            return version(PACKAGE)
        except PackageNotFoundError:  # running from a source tree, never installed
            return cls.FALLBACK

    @classmethod
    def install_command(cls, extra: str | None = None) -> str:
        target = f"{PACKAGE}[{extra}]" if extra else PACKAGE
        return f'uv add "{target} @ git+{REPO}@v{cls.installed()}"'
