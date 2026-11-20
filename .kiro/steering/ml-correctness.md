---
inclusion: always
---

# ML correctness steering — ML Factory 2.0

The platform's value is trustworthy numbers and human control. A change that violates one of these
rules is wrong even if its tests pass. They map to Correctness Properties 1–11 in
`.kiro/specs/ml-factory-2/design.md`.

## Deterministic first, LLM second

- Every number a user sees — statistic, metric, score, ranking, importance, cost, savings — comes
  from pandas/scikit-learn/the model libraries or the tracking service. The LLM never computes,
  estimates, re-derives or ranks a number.
- Every LLM response is validated against a versioned Pydantic schema before storage; references
  to nonexistent columns are dropped; invalid output is discarded.
- The LLM never mutates state, auto-excludes a feature, generates executable code, claims a job
  completed without verified status, or claims leakage is proven.

## No fitting on validation data

- Imputers, scalers, encoders, target encoding, learned bins/vocabularies and learned
  aggregations fit on the training fold only.
- Stateless row-wise derivations may precede the split only when they use neither the target nor
  information unavailable at prediction time.
- Never select features using the holdout test set. Ablations use identical validation folds.

## Human control

- Nothing is auto-excluded except the target (and explicit technical/security exclusions). Every
  finding is a warning with evidence; the user keeps, excludes, investigates, accepts risk, or
  compares with/without.
- Feature approval and feature selection are distinct.

## Leakage investigation (highest priority)

- Inherited deterministic rules and association calculations (|Pearson|, correlation ratio,
  Cramér's V) keep their tested behavior. New diagnostics are new rules with new stable ids.
- Association is not evidence of leakage. The LLM explains suspicion, evidence, and what would
  resolve it — never as verified fact.
- Pre- and post-training findings are separate artifacts; neither overwrites the other.
- Business context is captured once as a plain-language scenario; leakage investigation is on by
  default.

## Feature engineering

- Typed specifications from a closed vocabulary (ratio, difference, date difference, map
  categories, is-missing), never code. New operators only if validatable and safe, with tests.
- Never invent nonexistent history. Every proposal carries an interpretable hypothesis.

## Reproducibility

- Configuration is frozen to S3 before any job is submitted.
- Two SageMaker runs of the same frozen config (same image digest) must agree on the primary
  metric within 1e-6 absolute or 0.1% relative. No bitwise equality is required. Runs resumed
  from a Spot checkpoint are flagged and excluded from this guarantee.
- Only XGBoost and CatBoost are resumable; sklearn models restart on interruption. Never claim
  otherwise.
