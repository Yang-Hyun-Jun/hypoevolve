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

Write Python code that evaluates the measurable ELG hypothesis on the provided dataset and returns exactly one Python dictionary matching the required output contract from `evaluate_hypothesis(...)`.

Return only raw Python source for `candidate.py`.
Do not use markdown fences.
Do not add explanations, notes, or example usage.
Do not call the function at module scope.

Treat the column names shown in Column Specifications as exact, case-sensitive names.
Use only those exact dataset column names unless you explicitly create a derived column earlier in the function.
Do not assume transformed columns already exist in the input dataframe.
If you need a transformed signal, compute it from the provided base columns first and then reference the derived column you created.

If the measurable ELG contains parameter-slot notation such as `{RET_WINDOW}`, `{HORIZON}`, `{NEG_Z_THRESHOLD}`, or `{POS_Z_THRESHOLD}`, treat those as evaluator parameters, assign reasonable general-purpose values, use them consistently in the computation, and return the actual values used in `used_parameters`.
