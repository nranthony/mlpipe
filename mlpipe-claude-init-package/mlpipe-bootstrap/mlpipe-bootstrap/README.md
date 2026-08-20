# mlpipe bootstrap

Design + goals for a local, modular, agent-operated MLOps pipeline.
Produced from a design conversation; DESIGN.md is authoritative.

## Getting started
1. `git init`, commit this bundle as-is.
2. Install: `uv venv && uv pip install -e ".[dev]"` (add extras per cycle:
   `.[data]` for cycle 1, `.[ml]` cycle 6, `.[track]` cycle 7, `.[torch]` cycle 8).
3. Open Claude Code in the repo root and say:
   "Read DESIGN.md and CLAUDE.md, then execute goals/cycle0_spine.md."
4. One cycle per session/branch. Definition of done is in CLAUDE.md.

## Layout
- DESIGN.md — the system design (blocks, spine, ports, swappability, decisions)
- CLAUDE.md — agent operating rules
- goals/ — one goal file per build cycle + STATUS.md
- src/mlpipe/core/interfaces.py — authoritative port signatures (cycle 0 implements)
- TEMPLATE_OWNERSHIP.md — zones for later Copier template extraction
