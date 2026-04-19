# HypoEvolve Import & Interface Boundary Policy

This document defines the current boundary policy that should guide future
cleanup and restructuring work.

The goal is not to redesign directories yet. The goal is to make the current
roles and dependency directions explicit first, so later refactors can stay
behavior-preserving.

## 1. Current layer map

### 1.1 Public entry surfaces

- `hypoevolve/__init__.py:1-73`
  - owns the advertised package-root API via `__all__`
  - may re-export selected symbols from deeper modules
- `hypoevolve/cli.py:1-140`
  - owns Click command registration and user-facing command behavior

### 1.2 Application orchestration

- `hypoevolve/controller.py:1-140`
  - owns run orchestration, search-loop control flow, and the final `RunResult`
  - coordinates evaluator, worker, artifact, parser, archive, and runtime seams

### 1.3 Integration / runtime adapters

- `hypoevolve/evaluator.py:1-140`
  - defines the evaluator protocol and LLM-driven evaluator implementation
- `hypoevolve/workers.py:1-140`
  - defines worker task/result payloads and worker-side execution flow
- `hypoevolve/artifacts.py:1-140`
  - owns high-level run artifact assembly and recording
- `hypoevolve/runtime.py:1-84`
  - owns low-level file writing helpers for run directories and JSON payloads
- `hypoevolve/config.py:1-140`
  - owns config dataclasses and config-file loading

### 1.4 Domain / search support

Examples include:
- `hypoevolve/archive.py`
- `hypoevolve/mutation.py`
- `hypoevolve/parser.py`
- `hypoevolve/dataset.py`
- `hypoevolve/hypo/*`
- `hypoevolve/elg/*`

These modules hold core search semantics, ELG structure, parsing, mutation,
archive behavior, and dataset access rules.

## 2. Boundary rules

### Rule A — `hypoevolve.__init__` is the advertised root surface

The supported package-root API is the explicit `__all__` list in
`hypoevolve/__init__.py:49-73`.

Anything imported into `hypoevolve.__init__` but omitted from `__all__` should
be treated as a compatibility affordance or internal convenience, not as the
default advertised star-import surface.

### Rule B — production modules must not import the package root

Internal production code should import concrete modules directly, not
`hypoevolve` root exports.

Why:
- avoids circular public-surface coupling
- makes later export-policy cleanup safer
- keeps internal dependencies explicit

### Rule C — production modules must not import `hypoevolve.cli`

`hypoevolve.cli` is the command surface.
Other production modules must remain usable without importing CLI registration
code or command formatting helpers.

### Rule D — orchestration depends downward, not upward

`hypoevolve/controller.py` may coordinate archive, parser, evaluator, worker,
runtime, and artifact helpers, but those lower-level helpers should not import
the controller or CLI.

### Rule E — runtime writers stay low-level

`hypoevolve/runtime.py:14-84` should remain responsible only for raw run-dir
and JSON file writes.

Higher-level assembly, caching, report generation, and score-history semantics
belong in `hypoevolve/artifacts.py:23-140` and `hypoevolve/reporting.py`.

## 3. Practical interpretation for future work

- Before moving files, clarify interfaces and policy first.
- When narrowing exports, update tests that snapshot `__all__`.
- When adding new modules, choose the layer first, then import only downward.
- If a module needs symbols from `hypoevolve.cli` or package-root re-exports,
  that is a boundary smell and should be reviewed before merging.

## 4. What this policy does not do

This document does **not**:
- reorganize directories yet
- declare every currently reachable root attribute to be a stable public API
- change runtime behavior

It only freezes the minimum rules needed to keep the next structural work safe.
