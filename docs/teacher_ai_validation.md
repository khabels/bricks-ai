# Teacher AI Validation

Bricks AI can request candidate training data from external AI providers.

Provider roles:

- `Generator`: proposes training items
- `Validator`: checks correctness and relevance
- `Judge`: produces a final score or decision

A candidate item is accepted only if it passes configured thresholds.

```text
teacher_confidence >= min_teacher_confidence
validator_score >= consensus_threshold
question not empty
answer not empty
```

High-stakes topics should use stricter thresholds and should avoid personalized advice.
