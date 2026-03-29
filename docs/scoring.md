# Rule Scoring Methodology: Uplift × Coverage

## 1. Core Formula

We define:

- Precision: P(C | A ∧ B)
- Baseline: P(C)
- Coverage: P(A ∧ B)

### Uplift
uplift = P(C | A ∧ B) - P(C)

### Final Score
score = uplift × coverage

---

## 2. Intuition

This score represents:

> "How much better the outcome performs compared to baseline, weighted by how often the condition occurs."

To achieve a high score:

1. The condition must significantly improve the outcome (high uplift)
2. The condition must occur frequently enough (high coverage)

---

## 3. Deeper Interpretation

The formula can be rewritten as:

score = P(A ∧ B ∧ C) - P(A ∧ B)P(C)

This is equivalent to:

> Covariance between (A ∧ B) and C

Meaning:

- Positive score → positive relationship
- Zero → independent
- Negative → negative relationship

---

## 4. Why This Works

### 1. Handles baseline bias
High baseline values do not inflate the score.

### 2. Penalizes rare patterns
Rare but perfect rules are not overvalued.

### 3. Balances accuracy and applicability
It naturally balances precision and coverage.

---

## 5. Comparison to F1 Score

| Concept | F1 Score | This Method |
|--------|----------|------------|
| Precision | Yes | Yes |
| Recall | Yes | Approx (coverage) |
| Baseline correction | No | Yes |

This method is stricter than F1 because it removes baseline effects.

---

## 6. Interpretation

- score > 0 → predictive relationship
- score = 0 → no relationship
- score < 0 → inverse relationship

---

## 7. When It Works Best

- High baseline environments (e.g. finance)
- Noisy datasets
- Binary outcome problems
- Rule-based systems

---

## 8. Limitations

1. Does not capture variance
2. Ignores temporal structure
3. Only captures linear effect

---

## 9. Recommended Extensions

- Bootstrap confidence intervals
- Statistical significance tests
- Time-series validation

---

## 10. Key Takeaway

> A good hypothesis is not one that is often correct,
> but one that performs meaningfully better than baseline and occurs frequently.
