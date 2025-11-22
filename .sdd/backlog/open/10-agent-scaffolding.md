# Ticket 10: Agent Scaffolding & CLI Integration

**Goal**: Create the basic structure for the GitHub Discovery Agent and integrate it into the `ohmyrepos` CLI.

## Requirements
-   Create `src/agent/` package structure.
-   Create `src/agent/discovery.py` with a stub `discover` function.
-   Update `src/cli.py` to add `agent` command group and `discover` subcommand.
-   Ensure the CLI command `./run.sh agent discover --help` works.

## Implementation Details
-   **Files**:
    -   `src/agent/__init__.py`
    -   `src/agent/discovery.py`
    -   `src/cli.py` (modification)
-   **CLI Args**:
    -   `--category`: Optional filter.
    -   `--max-repos`: Default 20.

## Verification
-   Run `./run.sh agent discover --help` and verify output.
-   Run `./run.sh agent discover` and verify it prints a placeholder message.
