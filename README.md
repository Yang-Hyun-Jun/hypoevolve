# HypoEvolve

An experimental framework for **evolving hypotheses instead of code**.

HypoEvolve starts from a natural-language hypothesis, converts it into a structured intermediate representation called **ELG (Executable Logic Graph)**, mutates that structure, evaluates candidates, and keeps the best-scoring hypotheses over time.

> Current status: compact MVP with ELG core, archive, CLI, runtime persistence, and local worker support.

---

## What is HypoEvolve?

Most evolutionary systems mutate code, prompts, or numeric parameters. HypoEvolve explores a different object:

- **natural-language hypotheses**
- represented as **structured logic graphs (ELG)**
- mutated through **explicit structural operators**
- evaluated through a pluggable evaluator interface

The long-term goal is a data-driven hypothesis search system where LLMs can help:

- parse natural language into ELG
- suggest mutation directions
- critique candidate hypotheses
- assist evaluation over real datasets

---

## Current Features

### ELG core (`elg/`)
- atomic / logical / relation node model
- JSON serialization and deserialization
- normalization and fingerprinting
- structural metrics
- pretty and tree rendering
- immutable mutation primitives

### HypoEvolve app layer (`hypoevolve/`)
- natural-language input orchestration
- fallback NL → ELG parsing
- placeholder evaluator
- top-k archive with fingerprint dedup
- score-weighted parent sampling
- trace / checkpoint / best-result persistence
- subcommand CLI
- compact local worker support

---

## Repository Layout

```text
elg/            # hypothesis representation and mutation core
hypoevolve/     # app/runtime layer around ELG
tests/          # HypoEvolve app-layer tests
docs/           # project and comparison docs
```

---

## Installation

This repository is currently lightweight and Python-only.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

If you use Poetry:

```bash
poetry install
```

---

## Configuration

HypoEvolve uses a minimal config file:

```yaml
llm:
  api_key: EMPTY
  model: DeepSeek-R1-Distill-Qwen-14B
  api_base: http://127.0.0.1:8000/v1
  temperature: 0.2
  max_tokens: 2000
parser:
  retries: 1
  mode: fallback
search:
  iterations: 5
  random_seed: 42
archive:
  coverage_bins:
    - 0.05
    - 0.15
    - 0.30
  complexity_bins:
    - 3
    - 5
    - 8
output:
  base_dir: .hypoevolve/runs
logging:
  level: INFO
workers:
  enabled: false
  count: 1
```

If no config file is provided, HypoEvolve falls back to built-in runtime defaults.
The checked-in `hypoevolve.yaml` is the current **example starting point**, not the
implicit default loaded when the file is absent.

Recommended usage is to switch providers entirely in `hypoevolve.yaml`:

- local vLLM: set `api_base: http://127.0.0.1:8000/v1`, your local `model`, and
  keep `api_key: EMPTY` (or any other dummy string)
- OpenRouter or other hosted providers: replace `api_base`, `model`, and
  `api_key` in the same file

When you do pass `hypoevolve.yaml`, `api_key` from that file is used first, so you
do not need environment variables for normal provider switching.

---

## CLI

The CLI entrypoint is:

```bash
hypoevolve <subcommand>
```

### `run`
Run a HypoEvolve iteration loop from a natural-language hypothesis.

```bash
hypoevolve run "if signal A stays elevated then outcome B becomes more likely"
```

Use a custom config:

```bash
hypoevolve run "if signal A stays elevated then outcome B becomes more likely" --config hypoevolve.yaml
```

Override worker count:

```bash
hypoevolve run "if signal A stays elevated then outcome B becomes more likely" --workers 1
```

### `render`
Render a natural-language hypothesis using the fallback parser.

```bash
hypoevolve render "if signal A stays elevated then outcome B becomes more likely"
```

Tree mode:

```bash
hypoevolve render "if signal A stays elevated then outcome B becomes more likely" --tree
```

### `inspect`
Inspect a saved `best.json` or `checkpoint.json` file.

```bash
hypoevolve inspect .hypoevolve/runs/<run-id>/best.json
```

### `runs status`
Inspect one run by run id.

```bash
hypoevolve runs status <run-id>
hypoevolve runs status <run-id> --json
```

### `runs report`
Return or regenerate the markdown report for one run id.

```bash
hypoevolve runs report <run-id>
hypoevolve runs report <run-id> --json
```

### `doctor`
Show environment and config diagnostics.

```bash
hypoevolve doctor
```

---

## Runtime Output

Each run writes local files under:

```text
.hypoevolve/runs/<run-id>/
```

Current MVP outputs:

- `trace.jsonl`
- `checkpoint.json`
- `best.json`
- `artifacts/`

---

## Example: ELG rendering

```python
from elg import (
    AtomicNode,
    LogicalNode,
    RelationNode,
    Hypothesis,
    render_pretty,
    render_tree,
)

hypothesis = Hypothesis(
    root=RelationNode(
        "IMPLIES",
        [
            LogicalNode("AND", [
                AtomicNode("FEATURE_A_LEVEL > THRESHOLD_X"),
                AtomicNode("FEATURE_B_LEVEL > MOVING_AVERAGE_20"),
            ]),
            AtomicNode("OUTCOME_C_AT_NEXT_STEP == True"),
        ],
    )
)

print(render_pretty(hypothesis))
print(render_tree(hypothesis))
```

---

## Testing

Run HypoEvolve app-layer tests:

```bash
python -m unittest discover -s tests -p 'test_hypoevolve_*.py'
```

Run ELG regression tests:

```bash
python -m unittest discover -s openevolve/tests -p 'test_elg_*.py'
```

---

## Current Limitations

This is still an MVP / research-stage system.

Not implemented yet:
- real data-driven evaluator logic
- strong LLM-guided parser/evaluator integration
- islands / migration
- distributed runtime
- production-grade experiment management

The current evaluator is intentionally a placeholder.

---

## Design Notes

HypoEvolve does **not** try to clone OpenEvolve wholesale.

Instead, it reuses the useful high-level ideas:
- iterative search
- archive / best tracking
- mutation + evaluation loop
- eventual LLM-guided search

while replacing the core search object:
- **OpenEvolve → code/programs**
- **HypoEvolve → hypotheses/logic graphs**

See:
- `docs/openevolve.md`
- `docs/openevolve-vs-hypoevolve.md`

---

## Roadmap

Short-term:
- stronger NL → ELG parsing
- real evaluator integration
- better worker path testing outside sandbox constraints
- worker-aware runtime and UX polish

Mid-term:
- LLM-guided mutation proposals
- richer archive and diversity mechanisms
- stronger dataset/evidence interfaces

---

## Contributing

This project is still rapidly evolving. The cleanest way to contribute is to keep changes:
- compact
- well-tested
- structurally aligned with the ELG-first architecture

---

## Status

HypoEvolve is currently at:

- **ELG core** ✅
- **HypoEvolve MVP app layer** ✅
- **Local worker support** ✅
- **Real evaluator logic** ⏳
- **LLM-guided hypothesis loop** ⏳
