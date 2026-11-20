# Implementation Plan: Additional learners and ensembles

## Overview

Python, following `design.md`. The core insight: a trained model is always one `ModelPlugin`
producing one `ModelVersion` through `train_model`, so new learners are just plugins and an ensemble
is a plugin whose estimator embeds other plugins' estimators. Build bottom-up and phased: phase 1 is
the built-in learners and voting (no new dependency); phase 2 is stacking (gated on CV + the
model-version store) and the open-catalogue installed learners (gated on the stock-image install).
Every inherited-file change is additive and followed by a manifest update. A no-ensemble request
stays byte-for-byte today's. Commands: `.venv\Scripts\python -m pytest -q -p no:cacheprovider`,
`.venv\Scripts\python -m ruff check .`, `.venv\Scripts\python -m ruff format --check .`.

## Task Dependency Graph

```
1 built-in learners (NEW: boosting, bagging_trees) + registry (inherited) + PBT parity
└─▶ 2 ensemble contracts (NEW: contracts/ensemble.py) + config/model fields (inherited)
     └─▶ 3 ensemble plugin (NEW: models/ensemble.py) + get_plugin resolution (inherited)
          └─▶ 4 service: validate/guard/freeze/expand voting; request field (inherited)
               └─▶ 5 checkpoint
                    └─▶ 6 stacking: gate on CV, fit-cost guard, meta-learner default (service)
                         └─▶ 7 open catalogue: lightgbm plugin + installed-learner gating (service)
                              └─▶ 8 surface ensembles in start_training (MCP + API)
                                   └─▶ 9 end-to-end + backward-compat + golden tests
                                        └─▶ 10 final checkpoint
```

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "depends_on": [] },
    { "wave": 2, "tasks": ["2"], "depends_on": ["1"] },
    { "wave": 3, "tasks": ["3"], "depends_on": ["2"] },
    { "wave": 4, "tasks": ["4"], "depends_on": ["3"] },
    { "wave": 5, "tasks": ["5"], "depends_on": ["4"] },
    { "wave": 6, "tasks": ["6"], "depends_on": ["5"] },
    { "wave": 7, "tasks": ["7"], "depends_on": ["6"] },
    { "wave": 8, "tasks": ["8"], "depends_on": ["7"] },
    { "wave": 9, "tasks": ["9"], "depends_on": ["8"] },
    { "wave": 10, "tasks": ["10"], "depends_on": ["9"] }
  ]
}
```

## Tasks

- [x] 1. Built-in bagging and boosting learners (design C1; NEW modules, registry inherited)
  - `ml_engine/models/boosting.py` (NEW): `HistGradientBoostingClassifierPlugin`/`...Regressor`
    (sklearn `HistGradientBoosting*`), `AdaBoostClassifierPlugin`/`...Regressor`
  - `ml_engine/models/bagging_trees.py` (NEW): `BaggingClassifierPlugin`/`...Regressor`,
    `ExtraTreesClassifierPlugin`/`...Regressor`
  - Each declares `supported_problem_types`, `library="scikit-learn"`, `library_version`,
    conservative seeded `default_params`, `parameter_space` (so grid search works, Req 1.4),
    class-weighting where the estimator supports it; rely on base `feature_importance` with the
    permutation fallback for those exposing none (Req 1.6)
  - `ml_engine/models/registry.py`: register all eight plugins in the loop (inherited, additive);
    manifest update + justification; regenerate
  - Tests: per learner, register/available/build/train/predict/importance on a tiny fixture
    (Property 1); each imports only scikit-learn (Property 6)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 8.2_

- [x] 2. Ensemble contracts and the frozen config/model fields (design C3; inherited: config, model)
  - `ml_engine/contracts/ensemble.py` (NEW): `EnsembleType` (`voting`/`stacking`), `VotingRule`
    (`soft`/`hard`), `BaseLearnerSpec` (name + params + optional weight), `EnsembleSpec` (type, base
    learners min 2, voting_rule, meta_learner, derived `ensemble_id`), `EnsembleLineage` (base
    version ids). All versioned
  - `ml_engine/contracts/config.py`: add `ExperimentConfig.ensembles: list[EnsembleSpec]` (optional,
    excluded when empty so non-ensemble configs stay byte-identical, Req 7.2)
  - `ml_engine/contracts/model.py`: add optional `ModelMetadata.ensemble_lineage` (excluded when
    unset) for base-version lineage (Req 6.4)
  - Register `contracts/ensemble.py` as NEW; manifest updates + justifications for config.py,
    model.py; regenerate
  - Tests: every new contract round-trips; min-2 base learners enforced; `ensemble_id` stable and
    distinct for distinct specs; empty `ensembles` excluded from the serialised config
  - _Requirements: 7.1, 7.2_

- [x] 3. The ensemble plugin and its resolution (design C4; NEW: models/ensemble.py; registry inherited)
  - `ml_engine/models/ensemble.py` (NEW): `EnsemblePlugin(ModelPlugin)` built from an `EnsembleSpec`:
    `build` assembles `VotingClassifier/Regressor` (rule + weights) or `StackingClassifier/Regressor`
    (final_estimator = meta-learner, cv=K) over the base plugins' `build()` estimators;
    `preprocessing_strategy='dense_numeric'`; `feature_importance` returns None (permutation path);
    `is_available()` is the conjunction of base learners' availability
  - `ml_engine/models/registry.py`: the training job builds the `EnsemblePlugin` from the spec
    carried on its `ModelSpec` (keeping `get_plugin` pure for plain/grid names); add a small
    `build_ensemble_plugin(spec)` helper here so resolution has one home (inherited, additive);
    manifest update + justification; regenerate
  - Tests: a voting `EnsemblePlugin` builds a sklearn VotingClassifier over two base estimators;
    a stacking plugin builds a StackingClassifier with cv=K; `is_available` conjunction holds
  - _Requirements: 3.2, 3.5, 4.1, 4.5_

- [x] 4. Service: validate/guard/freeze/expand voting; request field (design C5, C6; inherited)
  - `backend/schemas.py`: `EnsembleRequest` (type, base_learners name+params+weight, voting_rule,
    meta_learner) and `TrainingRequest.ensembles: list[EnsembleRequest]` (absent = today, Req 7.2)
  - `backend/services/experiments.py`: validate each ensemble (≥2 base learners, each registered and
    supporting the problem type), freeze into `ExperimentConfig.ensembles`, expand each into one
    `ModelSpec(name=ensemble_id, params={frozen spec})` appended to `config.models`; record the
    ensemble definition + approver in the audit trail (Req 5.4). Voting costs like one model in the
    guards
  - Manifest updates + justifications for schemas.py, experiments.py if hash-tracked; regenerate
  - Tests: a voting request freezes an EnsembleSpec and adds one ModelSpec; <2 base learners refused;
    an unknown base learner refused naming it; a no-ensemble request adds nothing
  - _Requirements: 3.1, 3.4, 3.6, 5.2, 5.4, 6.1, 7.2_

- [x] 5. Checkpoint - Ensure all tests pass
  - `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `ruff check .`, `ruff format --check .`;
    confirm `tests/test_inherited_manifest.py` passes with the regenerated manifest; ask the user
    if questions arise

- [x] 6. Stacking: gate on CV, fit-cost guard, meta-learner default (design C4, C5)
  - `backend/services/experiments.py`: refuse a stacking ensemble unless `evaluation_mode='cv'`
    (Req 4.3) with "stacking requires CV_Mode"; default the meta-learner (logistic regression /
    ridge by problem type, Req 4.1/open q 2) when none is given; feed a stacking ensemble's
    `base × (K+1) + 1` fits into the CV fit-count ceiling and budget/concurrency guards (Req 4.7),
    refused with the numbers
  - Tests: stacking refused in Single_Split_Mode; the default meta-learner is applied; an
    over-ceiling stacking definition is refused with the counts before archiving (Property 5)
  - _Requirements: 4.1, 4.3, 4.7_

- [x] 7. Open catalogue: installed-learner plugin + gating (design C2; NEW: lightgbm_plugin.py)
  - `ml_engine/models/lightgbm_plugin.py` (NEW): LightGBM classifier + regressor with a lazy import
    and an `is_available()` returning `(False, "add 'lightgbm' to this experiment's requirements")`
    on ImportError; register it (and re-enable catboost registration) so the catalogue shows them as
    unavailable until installed (Req 2.1, 2.2)
  - `backend/services/experiments.py`: refuse selecting an installed learner (base, meta, or single)
    whose package is absent from the approved `requirements`, naming the missing package (Req 2.3);
    reuse the requirements name-normalisation
  - Register `lightgbm_plugin.py` as NEW; manifest update for registry.py; regenerate
  - Tests: the lightgbm plugin reports unavailable (not crash) when the library is absent; selecting
    it without the package in `requirements` is refused naming it; with the package present in
    `requirements` the submission is accepted (Property 2)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 7.5_

- [x] 8. Surface ensembles in start_training (design C6; inherited: MCP + API)
  - `mcp_server/server.py` + `backend/api/routes/training.py`: accept `ensembles` and echo the
    resolved ensembles (type, base-learner names, rule/meta, K) read back from the frozen config,
    beside the grid/treatment echoes; the LLM proposes, the user submits (Req 5.1-5.3, B12)
  - A `resolved_ensembles` view helper in `backend/api/views.py` (and the MCP twin), like
    `resolved_treatment`
  - Tests (in-process client / route): the echoed response lists the ensembles and their base learners
  - _Requirements: 5.1, 5.3, 6.2_

- [x] 9. End-to-end, backward-compat and golden tests (design Testing Strategy)
  - End-to-end offline (InlineOrchestrator): a voting ensemble of two learners under single split —
    each member fit on training rows, combined by the rule, one distinct loadable Model_Version
    ranked beside the members (Property 3); a stacking ensemble under CV_Mode — leak-free OOF
    meta-features, base learners refit on the full training set, one loadable Model_Version
    (Property 4); a per-ensemble failure leaves the others intact (Req 6.3)
  - Backward compatibility: a no-ensemble request is byte-identical to today (Property 6, golden)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.3, 8.4, 8.5_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `.venv\Scripts\python -m ruff check .`,
    `.venv\Scripts\python -m ruff format --check .`; confirm `tests/test_inherited_manifest.py`
    and `tests/test_architecture_boundaries.py` pass; ensure all pass; ask the user if questions arise
  - Result: ruff check + format clean; manifest + architecture-boundary tests pass; full suite green

## Notes

- A new learner is just a `ModelPlugin`; an ensemble is a plugin whose estimator embeds other
  plugins' estimators. Orchestration, training (single + CV), evaluation/ranking and the
  model-version store are reused with no change.
- scikit-learn's `Voting*`/`Stacking*` already fit base estimators on the given training rows (per
  fold under CV) and `Stacking*` builds leak-free OOF meta-features and refits bases on the full
  set — so fold-safety, OOF and refit (Req 3.5, 4.2, 4.5) come from the unchanged `train_model` path.
- Installed-library imports are lazy, gated by `is_available()`, so the offline suite and a
  no-`requirements` run never need the package (Req 7.5, 8.5).
- Property tests use Hypothesis (`max_examples=100`, `deadline=None`, `derandomize=True`), tagged
  `# Feature: ensembles-and-learners, Property N`.
