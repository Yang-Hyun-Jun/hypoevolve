# Current Parent Hypothesis (Natural Language)

{{PARENT_HYPOTHESIS_NL}}

# Current Parent Hypothesis (Measurable ELG)

{{PARENT_HYPOTHESIS_MEASURABLE}}


# Task

Generate a new child ELG hypothesis as a local exploratory mutation of the parent hypothesis.
Instead, prefer a valid, measurable, local mutation that explores a less-tried direction.

Return JSON only with this exact shape:

{
  "child_hypothesis": <ELG root node JSON>,
  "mutation_summary": "<what mutation-style changes were applied and where>"
}
