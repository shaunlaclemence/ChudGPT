# ChudGPT

Python library. Rotates free-tier provider API keys behind one client.

## Junior developer lessons: read these on EVERY prompt

Before writing or editing anything in this repo, read these lesson files under
`~/.claude/skills/junior-developer/lessons/`. Not once per session. Every prompt.
Re-read before each writing pass, including when continuing work already in progress.

| Lesson | Rule in one line |
| --- | --- |
| `minimal-comments.md` | Default to NO docstring and no comment. Single line only, only where naming does not carry it. Python docstrings count as comments. |
| `human-language.md` | No em dashes anywhere, in code, prose, commits, or chat replies. Use a period, comma, colon, or parentheses. |
| `oop-not-functional.md` | Class-based OOP with inheritance is the default. No modules of free functions, no module-level state, deps injected via `__init__` onto `self`. |
| `too-much-oop.md` | Bounds the above. Decorators are plain functions with `functools.wraps`, one per service in its own file, flat `try/except` chain. Never a base class with a `translate()` override chain. |
| `python-app-structure.md` | Every Python app gets `repositories/ services/ schemas/ exceptions/ utils/`. The exceptions layer exposes a handler decorator that services are wrapped with. |
| `exception-hierarchy-pattern.md` | Exceptions carry message, error_code, service_code (`ServiceCode` enum), and the ORIGINAL error. Two inheritance axes giving `<service>-<error>` codes. |
| `logic-belongs-in-utils.md` | Services orchestrate, utils decide. Guards, validation, and patch building move to a `<Domain>Rules`/`Policy`/`Field` class. No pass-through aliases. |

`minimal-comments` and `human-language` are the two that get violated by drift: read
at the top of a task, then quietly abandoned several files later. Check the diff
against them before reporting work as done.

`too-much-oop` and `exception-hierarchy-pattern` look like they conflict. They do not.
Exception TYPES stay a class hierarchy. Exception HANDLERS are plain functions.

## Read when the task touches them

| Lesson | When |
| --- | --- |
| `git-conventions.md` | Any commit, branch, message, or PR. Starts with: never co-author a commit. |
| `git-dependency-consumption.md` | Packaging, versioning, tags, or how another app installs this one. This repo is consumed as a git dependency. |

## Never applies here

`deck-nav-uses-routes.md` is scoped to the beaver.ai desktop React app.

## Capturing corrections

When the senior dev corrects how something was done, record it per the skill before
moving on: update the existing lesson if one covers it, otherwise write
`lessons/<slug>.md` and append one line to `LESSONS.md`. Then say
`Recorded: <slug> - <the rule>.` and keep building.

## Project specifics

- Package layout lives under `src/chudgpt/`. Services in `services/`, their handlers in
  `services/exceptions/<service>.py`, exception types in `exceptions.py`.
- Public API is `chudgpt` (client and the types its methods take) and
  `chudgpt.exceptions` (every error). Everything else is internal.
- `secrets.json` holds real API keys and is gitignored. Never read it, print it, or
  commit it.
- Verify with `uv run pytest` and `uv run ruff check` before reporting done.
