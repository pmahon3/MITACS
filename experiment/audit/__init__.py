"""Applied-research-audit support package.

Two modules:

  * ``registry`` — read/write/verify ``notes/preregistrations/``
    entries. Hash-stamping via ``experiment._prov_core``; append-only
    convention; schema validation per the ``.claude/agents/*.md``
    specifications.

  * ``code_path`` — static inspection of a Python script to determine
    whether it calls production functions (whitelist below) or
    reimplements production logic. Feeds ``.claude/agents/code-path-auditor.md``.

Design: ``notes/seeds/applied_audit_workflow.md`` §9.
Threat model: ``memory/mitacs-session-lessons-corpus.md``.
"""
