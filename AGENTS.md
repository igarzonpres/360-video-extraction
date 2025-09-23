# Agent Working Agreement for This Project

This file documents how I (the Codex CLI agent) should work in this repo so my changes align with your preferences.

## Branching With Worktrees

- One feature per branch. Never work directly on `main` for features or fixes.
- Use Git worktrees to keep branches isolated in separate folders and enable quick context switching.

Recommended pattern (PowerShell):

```
# From the repository root
git fetch origin
$branch = "feature/<short-topic>"
git worktree add ../wt-$($branch -replace '/','-') -b $branch origin/main
```

When done and merged:

```
git worktree remove ../wt-<feature-branch-with-dashes>
git branch -d <feature-branch>
git worktree prune
```

Notes:
- Name worktree folders `wt-<branch>` to avoid confusion.
- Keep branches focused and short‑lived; open PRs early for visibility.

## Pull Requests

- Always open a PR from the feature branch into `main`.
- Prefer the GitHub CLI when available:

```
gh pr create -B main -H <feature-branch> -t "<concise title>" -b "<clear summary>"
```

- Titles: start with area prefix and intent (e.g., `feat(ui): …`, `fix(av): …`).
- Bodies: summarize the why, list the what, note risks, and how to validate. Use bullets. If relevant, reference tests added.
- Keep diffs focused; split unrelated changes into separate PRs.

## Planning & TODOs (Before Coding)

- Always produce a short TODO plan before writing code.
- Share the plan with the user and wait for explicit approval before implementing.
- Use the plan tool (update_plan) to track steps; keep items concise (5–7 words each).
- Show the plan up front, then implement exactly what’s approved.
- Update the plan as scope evolves; keep exactly one step in progress.
- Do not start file edits or commands until the user approves the TODO/plan.

## Tests‑First Workflow

- Write or update tests first to describe the desired behavior. Let them fail.
- Implement the minimal code to make the tests pass.
- Iterate until green, then refactor if needed.

Where to put tests:
- Unit/integration tests live under `tests/`.
- Prefer small, focused tests colocated by domain (e.g., operations → `tests/*operations*`).

How to run tests:

```
# Windows PowerShell (from repo root)
$env:PYTHONPATH = 'src'
pytest -q

# Or with uv (if installed)
uv run pytest -q
```

Conventions:
- Use deterministic inputs and temporary directories (e.g., `tempfile`), and clean up afterward.
- For GUI changes, add minimal smoke tests if feasible (non‑interactive parts only). Do not attempt to test interactive Qt flows here.

## Coding & Commits

- Keep changes minimal and scoped to the task.
- Prefer small, readable functions; avoid unnecessary abstractions.
- Follow existing code style (imports, naming, logging).
- Commit early and often with meaningful messages; squash on merge if needed.

Examples:

```
feat(ui): enlarge images list and move controls
fix(ffmpeg): stream output and support cancel
chore(theme): remove dark theme switching UI
```

## Dependencies & Environment

- Do not add dependencies unless strictly necessary. Confirm before introducing new ones.
- Respect the existing tooling: `uv` for env management is preferred when used.

## Safety & Housekeeping

- Never remove or rewrite large files unless explicitly requested.
- Do not modify CI or release config without prior agreement.
- If a task requires destructive operations (rename/move many files), add a dry‑run mode or prompts where possible.

## Simplicity & Dead Code

- Prefer the simplest possible solution that clearly serves the user goal.
- Proactively look for and remove redundant, unused, or outdated code and assets.
- Avoid keeping commented‑out blocks or obsolete files; delete them once superseded.
- When refactoring, keep PRs focused and describe removals clearly in the body.

---

The agent should read and follow this AGENTS.md for any work performed in this repo. If something here conflicts with explicit user instructions in a session, the user’s instructions take precedence.
