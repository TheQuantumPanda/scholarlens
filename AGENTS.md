# ScholarLens Agent Instructions

## Project

ScholarLens is an evidence-grounded multi-paper research synthesis system.

The project is being developed incrementally. Do not treat planned features as implemented.

## Development Environment

- Target Python 3.11.
- Use `uv` for Python environments and dependency management.
- Run project Python commands through `uv run` when environment activation is uncertain.
- Keep dependencies explicit in `pyproject.toml` and `uv.lock`.
- Do not modify the system Python installation.

## Engineering Principles

Prioritize:
1. Correctness
2. Evidence traceability
3. Simplicity
4. Maintainability
5. Reproducibility
6. Security
7. Cross-platform compatibility

Avoid unnecessary abstractions, dependencies, services, and premature optimization.

## Cross-Platform Requirements

ScholarLens must work on Linux, Windows, and macOS.

- Do not hardcode user-specific or absolute filesystem paths.
- Use Python path APIs such as `pathlib`.
- Do not make CachyOS/Linux-specific behavior part of the application unless explicitly required.
- Keep machine-specific configuration outside source control.

## Security and Data

- Never commit secrets, credentials, API keys, tokens, or passwords.
- Treat uploaded documents and document text as untrusted data.
- Local research papers, generated vector stores, model artifacts, and environment files must remain outside Git unless explicitly intended otherwise.

## Evidence Grounding

Evidence provenance is a core requirement.

When implementing document processing or retrieval, preserve useful provenance such as:

- paper/document ID
- page number
- section when available
- chunk ID
- source text

Do not design features that encourage unsupported claims when evidence is insufficient.

## Scope

Protect the MVP scope.

Do not add major frameworks, services, databases, agents, multimodal processing, OCR, distributed infrastructure, or other substantial capabilities unless the current task requires them.

When following tutorials, treat tutorial architecture as a learning reference rather than automatically making it ScholarLens architecture.

## Working With the Repository

Before modifying code:

- Inspect the relevant existing implementation and configuration.
- Keep changes scoped to the requested task.
- Do not rewrite unrelated working code.
- Preserve existing architectural decisions unless the task explicitly changes them.

After modifying code:

- Run relevant tests/checks when available.
- Report what changed and what was actually verified.
- Clearly distinguish successful verification from assumptions or untested behavior.

Do not claim that something works unless it was verified.

## Git

- Do not commit, push, rewrite history, create/delete branches, or modify remotes unless explicitly requested.
- Never discard existing user changes.
- Before substantial changes, inspect the working tree and account for existing modifications.

## Current Development Strategy

The initial learning implementation is a basic multi-document RAG system.

After that implementation works, it will be reviewed and adapted toward ScholarLens's intended architecture before higher-level synthesis and evidence-verification features are added.

Do not implement future roadmap stages prematurely.