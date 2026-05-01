# HypoEvolve

An experimental framework for **evolving hypotheses instead of code**.

HypoEvolve starts from a natural-language hypothesis, converts it into a structured intermediate representation called **ELG (Executable Logic Graph)**, mutates that structure, evaluates candidates, and keeps the best-scoring hypotheses over time.

> Current status: layered architecture with ELG core, MAP-Elites archive, protocol-based skill system, CLI, runtime persistence, and parallel worker support.

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

### ELG core (`hypoevolve/elg/`)
- atomic / logical / relation node model
- JSON serialization and deserialization
- normalization and fingerprinting
- structural metrics
- pretty and tree rendering
- immutable mutation primitives

### Orchestration & Skills (`hypoevolve/core/`, `hypoevolve/skills/`)
- natural-language input orchestration with HookBus event system
- ELG compile skill (NL → ELG parsing)
- mutation skill with structural operators
- pluggable evaluation skill interface
- reporting skill for run summaries
- seed generation subsystem for random hypothesis bootstrapping

### Policies (`hypoevolve/policies/`)
- UCB-based parent selection within MAP-Elites cells
- protocol-based selection and stopping policies

### Memory & Archive (`hypoevolve/memory/`)
- MAP-Elites archive with fingerprint dedup and coverage/complexity binning
- artifact persistence (trace, checkpoint, best-result)

### Runtime (`hypoevolve/runtime/`)
- LLM client with retry and provider abstraction
- sandboxed code execution
- checkpoint persistence
- parallel worker support

### CLI (`hypoevolve/cli/`)
- subcommand CLI (`run`, `seed`, `render`, `inspect`, `doctor`)
- `runs` subgroup (`status`, `report`)
- styled output and diagnostics

---

## Repository Layout

```text
hypoevolve/
  core/           # orchestrator, config, HookBus event system
  elg/            # ELG representation, mutation, rendering, codecs
  skills/         # protocols, ELG compile, mutation, evaluation, reporting
    seed_generation/  # random seed-hypothesis generation subsystem
  policies/       # protocols, UCB selection, stopping policies
  context/        # protocols, prompt variable providers
  memory/         # MAP-Elites archive, artifact persistence
  runtime/        # LLM client, sandbox executor, checkpoint, workers
  data/           # dataset loader, YAML parser
  observability/  # structured logger
  cli/            # app, commands, runs subgroup, display helpers
  prompts/        # packaged markdown prompt templates
tests/            # 258 tests covering all layers
docs/             # project and comparison docs
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
  api_key: <your-api-key>
  api_base: https://openrouter.ai/api/v1
  model: deepseek/deepseek-v4-pro
  temperature: 0.8
  max_tokens: 10000
  retries: 1
  retry_delay: 1.0
parser:
  retries: 2
evaluator:
  dataset_schema_path: dataset.yaml
  parameters:
    window: 12
    horizon: 1
  seed: 42
search:
  iterations: 100
  steering_retries: 3
  random_steering_prob: 0.3
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
  parent_sampling_mode: map_elites_ucb
output:
  base_dir: .hypoevolve/runs
  top_k_evaluator_code_artifacts: 3
logging:
  level: INFO
workers:
  enabled: true
  count: 4
```

If no config file is provided, HypoEvolve falls back to built-in runtime defaults.
The checked-in `hypoevolve.yaml` is the current **example starting point**, not the
implicit default loaded when the file is absent.

Recommended usage is to switch providers entirely in `hypoevolve.yaml`:

- **OpenRouter**: set `api_base: https://openrouter.ai/api/v1`, your `model`
  (e.g. `deepseek/deepseek-v4-pro`), and your OpenRouter API key
- **local vLLM**: set `api_base: http://127.0.0.1:8000/v1`, your local `model`,
  and keep `api_key: EMPTY` (or any dummy string)
- **other hosted providers**: replace `api_base`, `model`, and `api_key`

When you pass `hypoevolve.yaml`, the `api_key` from that file is used first, so you
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
from hypoevolve.elg import (
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

`hypoevolve.elg` is the canonical import path for all ELG types and functions.

---

## Testing

Run all tests:

```bash
python -m pytest tests/
```

---

## Current Limitations

Not implemented yet:
- islands / migration across archive populations
- distributed runtime (beyond local parallel workers)
- production-grade experiment management

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
- stronger NL → ELG parsing via LLM
- richer evaluator integrations
- worker-aware runtime and UX polish

Mid-term:
- LLM-guided mutation proposals
- island/migration across archive populations
- stronger dataset/evidence interfaces
- distributed runtime support

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
- **Layered architecture with protocol abstractions** ✅
- **MAP-Elites archive with UCB selection** ✅
- **HookBus event system** ✅
- **Parallel worker support** ✅
- **CLI with runs management** ✅
- **Island/migration** ⏳
- **Distributed runtime** ⏳
