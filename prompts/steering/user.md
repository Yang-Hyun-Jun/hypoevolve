# Current Parent Hypothesis (Measurable ELG)

{{PARENT_HYPOTHESIS_MEASURABLE}}

# Current Parent Hypothesis (Natural Language)

{{PARENT_HYPOTHESIS_NL}}

# Current Evaluation Metrics

{{CURRENT_METRICS}}

# Metric Definitions

{{METRIC_DEFINITIONS}}

# Recent Mutation History

{{RECENT_HISTORY}}

# Top Archive Hypotheses

{{TOP_HYPOTHESES}}

# Task

Generate a new child ELG hypothesis as a local mutation of the parent hypothesis.

Return JSON only with this exact shape:

{
  "domain_reason": "<why this mutation is plausible from a domain-knowledge perspective>",
  "score_reason": "<why this mutation is promising from a score-improvement perspective>",
  "child_hypothesis": <ELG root node JSON>,
  "mutation_summary": "<what mutation-style changes were applied and where>"
}
