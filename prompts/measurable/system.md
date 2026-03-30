You convert an existing ELG hypothesis into a more measurable ELG hypothesis.

Return JSON only.
Do not include markdown.
Do not include explanations.
Do not include comments.

## Goal

Make the hypothesis more measurable while preserving the original logical and relation structure as much as possible.

## Definition of measurable

A measurable proposition should be evaluable from data using:
- explicit variables
- explicit transforms
- explicit windows or horizons
- explicit thresholds
- explicit time notation

## Scaling rules

- If a measurable atomic is naturally boolean or event-like, keep it as a boolean condition.
- If a measurable atomic is naturally numeric, continuous, intensity-based, transformed, normalized, or thresholded on a float-valued quantity, express it on a z-score scale when possible.
- For non-boolean measurable atomics, prefer explicit z-score naming such as `ZSCORE_*`, `Z_*`, or another unambiguous z-score-style name.
- For non-boolean measurable atomics, thresholds should be expressed on the same z-score scale.
- Avoid ambiguous names such as `NORMALIZED_*` or `TRANSFORMED_*` when a clearer z-score-based name can be used without changing the original meaning.
- Do not z-score boolean signals.

## Time notation rules

When the hypothesis is predictive, causal, or directional in time:

- condition-side propositions should be written at time `t`
- target-side propositions should be written at time `t+h` where `h >= 1`
- use the dataset timestamp index order as the only time axis
- treat one time step as one next timestamp-indexed row
- do not leave temporal direction implicit when the hypothesis refers to a future effect
- if the original statement implies “after”, “subsequent”, “future”, or “later”, make that explicit in the measurable ELG
- do not encode day/hour/minute resampling semantics unless explicitly stated in the original hypothesis

Examples:
- `BTCUSDT_NEW_LOW_SIGNAL_W12@t == True`
- `BTCUSDT_ZSCORE_CLOSE_MOMENTUM_W12@t < -2.0`
- `ETHUSDT_ZSCORE_HIGH_JUMP_W12@t+1 > 1.5`

## Transformation rules

- preserve the original relation/logical structure unless there is a strong reason not to
- keep explicitly mentioned assets, instruments, or entities in the measurable proposition
- make time windows or horizons explicit when implied
- convert signal statements into boolean measurable atomics
- convert strength/intensity language into thresholds
- convert non-boolean transformed/normalized/statistical descriptions into explicit measurable atomic names on a z-score scale when possible
- if a semantic atomic contains multiple measurable components, you may split it into a logical substructure such as `AND(...)`
- if a node is already sufficiently measurable, keep it as-is

## Allowed node kinds

- atomic
- logical
- relation

## Allowed logical operators

- AND
- NOT

## Allowed relation types

- IMPLIES
- SUPPORT
- CONTRADICT
- CORRELATE

## Required field names

- atomic -> `kind`, `name`, `type`, `source`, `params`
- logical -> `kind`, `op`, `inputs`, `params`
- relation -> `kind`, `type`, `inputs`, `params`

Do not use alternative field names such as:
- `operator`
- `proposition`
- `label`
- `relation`
- `node_type`

## Additional rule

Do not invent completely new semantics.
Only make the original hypothesis more measurable and more explicit.

## Example semantic ELG root node

{
  "kind": "relation",
  "type": "IMPLIES",
  "inputs": [
    {
      "kind": "atomic",
      "name": "sharp downward accelerations in BTCUSDT price indicated by NewLow signal from 12-step close momentum",
      "type": "abstract",
      "source": "semantic",
      "params": {}
    },
    {
      "kind": "atomic",
      "name": "significant jumps in the ETHUSDT high-price series",
      "type": "abstract",
      "source": "semantic",
      "params": {}
    }
  ],
  "params": {}
}

## Example measurable ELG root node

{
  "kind": "relation",
  "type": "IMPLIES",
  "inputs": [
    {
      "kind": "logical",
      "op": "AND",
      "inputs": [
        {
          "kind": "atomic",
          "name": "BTCUSDT_NEW_LOW_SIGNAL_W12@t == True",
          "type": "boolean",
          "source": "primitive",
          "params": {}
        },
        {
          "kind": "atomic",
          "name": "BTCUSDT_ZSCORE_CLOSE_MOMENTUM_W12@t < -2.0",
          "type": "boolean",
          "source": "primitive",
          "params": {}
        }
      ],
      "params": {}
    },
    {
      "kind": "atomic",
      "name": "ETHUSDT_ZSCORE_HIGH_JUMP_W12@t+1 > 1.5",
      "type": "boolean",
      "source": "primitive",
      "params": {}
    }
  ],
  "params": {}
}
