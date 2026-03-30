# Hypothesis (Readable)

{{HYPOTHESIS_PRETTY}}

# Dataset Description

{{DATASET_DESCRIPTION}}

# Dataframe Index (Not Column)

name: {{INDEX_NAME}}
dtype: {{INDEX_DTYPE}}

# Entities

{{ENTITIES}}

# Column Specifications

{{COLUMN_SPECS}}

# DatasetAccessor Interface

{{DATASET_ACCESSOR_DOC}}

# Task

Write Python code that evaluates the measurable ELG hypothesis on the provided dataset and prints exactly one JSON object matching the required output contract.

If the measurable ELG contains parameter-slot notation such as `{RET_WINDOW}`, `{HORIZON}`, `{NEG_Z_THRESHOLD}`, or `{POS_Z_THRESHOLD}`, treat those as evaluator parameters, assign reasonable general-purpose values, and use them consistently in the computation.
