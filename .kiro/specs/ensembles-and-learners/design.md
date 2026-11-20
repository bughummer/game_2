# Design: Additional learners and ensembles

## Overview

Three additions, each landing behind an interface the pipeline already has, so orchestration never
changes:

1. **Built-in learners** — new `ModelPlugin`s built only from the stock image's libraries
   (scikit-learn): `hist_gradient_boosting`, `adaboost`, `bagging`, `extra_trees`, each a
   classifier + regressor. They register in `ml_engine/models/registry.py` and flow through
   training, CV, evaluation, ranking, importance, grid search and validation untouched, because the
   plugin contract is the only integration point.

2. **Open catalogue** — a learner whose library is not in the base image (lightgbm, catboost)
   becomes usable when the experiment's `requirements` (stock-image B10) installed it. Its plugin
   ships with the platform and uses the existing `is_available()` gate to report "unavailable; add
   it to this experiment's requirements" until the library is present. The service refuses a
   submission that selects such a learner without the needed package in the approved `requirements`.

3. **Ensembles** — two combiners, each saved as its own `ModelVersion`:
   - **Voting** (phase 1): a fixed rule (soft/hard vote for classification, mean for regression). A
     `VotingPlugin` wraps scikit-learn's `VotingClassifier`/`VotingRegressor` around the chosen base
     learners. Nothing is learned on top, so it needs neither CV nor the model-version store.
   - **Stacking** (phase 2, gated): a `StackingPlugin` wraps `StackingClassifier`/`StackingRegressor`,
     whose meta-learner trains on the base learners' cross-validated out-of-fold predictions.
     Refused outside CV_Mode.

The central design fact, from reading the code: a trained model in this platform is always one
`ModelPlugin` producing one `ModelVersion` through `train_model`. An ensemble is therefore modelled
as **a plugin whose estimator happens to contain other plugins' estimators**, not as a new stage.
scikit-learn's own `VotingClassifier`/`StackingClassifier` already fit their base estimators on the
training rows they are given (per fold under CV, because the whole plugin is fit per fold), and
`Stacking*` already builds leak-free OOF meta-features via an internal K-fold and refits the base
estimators on the full training set. So the fold-safety and refit requirements (Req 3.5, 4.2, 4.5)
are satisfied by the same `train_model` path every other plugin uses — the ensemble plugin just
assembles the sklearn estimator from the approved base-learner specs.

When no ensemble is defined and only the existing learners are requested, every artifact is
byte-for-byte today's.

## Architecture

```
registry (additive):
  built-in plugins  ──▶ hist_gradient_boosting, adaboost, bagging, extra_trees (clf + reg)
  installed plugins ──▶ lightgbm, catboost (gated by is_available(); reachable via B10 install)
  ensemble plugins  ──▶ assembled on demand from an EnsembleSpec, not pre-registered

training request with ensembles:
  EnsembleSpec(type=voting|stacking, base_learners=[ModelSpec...], rule|meta_learner)
        │  (service validates + freezes into ExperimentConfig.ensembles)
        ▼
  ExperimentService expands each EnsembleSpec into one ModelSpec with an ensemble name
  (e.g. 'ensemble_voting__<hash>') whose params carry the frozen spec; these join config.models
        ▼
  training stage: get_plugin(name) returns an EnsemblePlugin built from the frozen spec; it
  constructs the sklearn Voting/Stacking estimator over the base plugins' build() outputs and
  trains through the unchanged train_model → one ModelVersion per ensemble
        ▼
  evaluation ranks the ensemble beside the single learners by Best_By_CV_Mean; nothing auto-selected
```

Built-in and installed learners need **no** service or job change beyond registration. Ensembles
need a spec contract, a plugin that reads it, a name→spec resolution at training, service-level
validation/guards, and request surfacing.

## Components and Interfaces

### C1. Built-in learner plugins (`ml_engine/models/*.py`, NEW modules; registry inherited)

- `ml_engine/models/boosting.py` (NEW): `HistGradientBoostingClassifierPlugin`/`...Regressor`
  (sklearn `HistGradientBoostingClassifier/Regressor`), `AdaBoostClassifierPlugin`/`...Regressor`.
- `ml_engine/models/bagging_trees.py` (NEW): `BaggingClassifierPlugin`/`...Regressor`,
  `ExtraTreesClassifierPlugin`/`...Regressor`.
- Each declares `supported_problem_types`, `library = "scikit-learn"`, `library_version`,
  `default_params` (conservative, seeded with `context.random_seed`), `parameter_space` (so grid
  search works, Req 1.4), class-weighting where the estimator supports it, and relies on the base
  `feature_importance` (native `feature_importances_`), falling back to permutation where a learner
  exposes none (Req 1.6 — already the platform's behaviour).
- `registry.py` gains these in the registration loop (inherited edit, additive, manifest).
  HistGradientBoosting uses the shared dense-numeric path for parity (open q 4).

### C2. Installed-learner plugins (`ml_engine/models/*.py`, NEW; registry inherited)

- `ml_engine/models/lightgbm_plugin.py` (NEW) and the existing `catboost_plugin.py` (kept on disk):
  each declares its `library` and an `is_available()` that tries to import the library and returns
  `(False, "add '<pkg>' to this experiment's requirements")` on `ImportError` (Req 2.1, 2.2).
  Imports of the optional library are **lazy** (inside `is_available`, `build`, `library_version`),
  never at module top level, so the offline suite and a no-`requirements` run never need the package
  (Req 7.5, 8.5).
- Registered in `registry.py` so they appear in the catalogue as unavailable until installed. The
  service (C5) refuses selecting an installed learner whose package is absent from the approved
  `requirements` (Req 2.3), naming it.
- An installed learner flows through every stage exactly like a built-in one (Req 2.4); its
  reproducibility record is the saved `pip freeze` on its Model_Version (Req 2.6, stock-image B11).

### C3. Ensemble contracts (`ml_engine/contracts/ensemble.py`, NEW)

- `EnsembleType` enum: `voting`, `stacking`.
- `VotingRule` enum: `soft`, `hard` (classification); regression ignores it (mean).
- `BaseLearnerSpec` (versioned): a base learner's `name` + `params` (an embedded `ModelSpec`-shaped
  choice) and an optional `weight` (voting, default equal).
- `EnsembleSpec` (versioned): `type`, `base_learners: list[BaseLearnerSpec]` (min 2, Req 3.1/4.1),
  `voting_rule` (voting only), `meta_learner: ModelSpec | None` (stacking only; default logistic
  regression for classification, ridge for regression, open q 2), and a derived, stable
  `ensemble_id` (`ensemble_<type>__<hash8>` over the sorted spec) so each ensemble has its own
  artifacts, mirroring the grid-point id scheme.
- `EnsembleLineage` (versioned): the base-learner version ids an ensemble combines (Req 6.4),
  recorded on the ensemble's metadata for traceability.
- Frozen into `ExperimentConfig.ensembles: list[EnsembleSpec]` (optional, omitted when empty so
  non-ensemble configs stay byte-identical, Req 7.2).

### C4. The ensemble plugin (`ml_engine/models/ensemble.py`, NEW; registry resolution inherited)

- `EnsemblePlugin(ModelPlugin)` built from a frozen `EnsembleSpec`:
  - `build(params, context)` constructs the sklearn estimator: `VotingClassifier`/`VotingRegressor`
    (with `voting='soft'|'hard'` and weights) or `StackingClassifier`/`StackingRegressor` (with
    `final_estimator` = the meta-learner's built estimator and `cv=K`), whose `estimators` are
    `(name, base_plugin.build(base_params, context))` for each `BaseLearnerSpec`.
  - `preprocessing_strategy` is `dense_numeric` (every base learner here consumes the dense matrix;
    an ensemble mixing a native-categorical base is out of scope for phase 1).
  - `feature_importance` returns `None` → the permutation path provides importance (Req 1.6),
    since a voting/stacking estimator has no single native importance.
  - `is_available()` is the conjunction of its base learners' availability, so an ensemble naming an
    uninstalled learner reports the same clear reason.
- **Resolution**: `registry.get_plugin(name)` already maps a `<base>__<hash>` name to a base
  plugin. Ensemble names use an `ensemble_<type>__<hash>` prefix; `get_plugin` is extended so an
  ensemble name resolves to an `EnsemblePlugin` constructed from the matching `EnsembleSpec` found in
  the frozen `ExperimentConfig` (passed via a thread-safe lookup the training job sets up, or — the
  chosen approach — the ensemble spec is carried in the `ModelSpec.params`, so the plugin is built
  from the spec at `get_plugin` time without reading global state). The training job builds the
  `EnsemblePlugin` from the spec attached to its `ModelSpec`, keeping `get_plugin` pure for plain and
  grid names.

### C5. Service: validate, guard, freeze, expand (`backend/services/experiments.py`, inherited)

- `TrainingRequest` gains optional `ensembles: list[EnsembleRequest]` (C6). `ExperimentService`:
  - validates each ensemble: ≥2 base learners (Req 3.1/4.1), every base learner and the meta-learner
    are registered and support the problem type, an **installed** base/meta learner's package is in
    the approved `requirements` (Req 2.3), and **stacking is refused unless `evaluation_mode='cv'`**
    (Req 4.3) with a clear "stacking requires CV_Mode".
  - freezes the validated `EnsembleSpec`s into `ExperimentConfig.ensembles` and expands each into one
    `ModelSpec(name=ensemble_id, params={... frozen spec ...})` appended to `config.models`, so the
    existing per-model training/evaluation/model-version machinery runs them unchanged (Req 6.1).
  - feeds the fit cost into the existing guards: a voting ensemble costs like one model (its K+1 fold
    fits under CV); a **stacking** ensemble costs `base_count × (K + 1)` (OOF + final refit) + 1
    meta-fit, added to the CV `model_count × (K+1)` ceiling and the budget/concurrency guards
    (Req 4.7), refused with the numbers if a ceiling is exceeded.
  - records the ensemble definition and approver in the audit trail (Req 5.4).

### C6. Request + surfacing (`backend/schemas.py`, `mcp_server`, `backend/api`)

- `EnsembleRequest` (API/MCP shape): `type`, `base_learners` (name + params + weight), `voting_rule`,
  `meta_learner`. `TrainingRequest.ensembles: list[EnsembleRequest]` (absent = today, Req 7.2).
- `start_training` echoes the resolved ensembles (type, base-learner names, rule/meta, K) read back
  from the frozen config, beside the grid/treatment echoes. The LLM may propose; the user submits
  (Req 5.1-5.3, B12). No winner is chosen by the platform (Req 5.3, 6-ranking unchanged).

## Data Models

New: `ml_engine/contracts/ensemble.py` (`EnsembleType`, `VotingRule`, `BaseLearnerSpec`,
`EnsembleSpec`, `EnsembleLineage`). `ExperimentConfig.ensembles: list[EnsembleSpec]` (optional,
empty-excluded). `ModelMetadata` gains an optional `ensemble_lineage: EnsembleLineage | None`
(excluded when unset) so an ensemble version references its base versions (Req 6.4). New layout:
none required beyond the existing `models/<name>/` tree — an ensemble is a model, so it reuses every
model path. The sklearn estimator (which embeds the fitted base estimators) is the one saved
artifact, so loading an ensemble version loads its base models with it (Req 6.2).

## Correctness Properties

### Property 1: plugin parity
Every new built-in learner registers, reports availability, builds with defaults, trains, predicts
and yields importance (native or permutation) on a tiny fixture, with no orchestration branch on its
name.
**Validates: Requirements 1.1, 1.2, 1.3, 8.2**

### Property 2: installed-learner gating
An installed-learner plugin reports `is_available() == (False, reason)` when its library is absent,
never raising `ImportError` at import or descriptor time; selecting it without the package in
`requirements` is refused with the package named.
**Validates: Requirements 2.1, 2.2, 2.3, 7.5, 8.5**

### Property 3: voting fold-safety and independence
A voting ensemble fits each base learner on the training rows only (per fold under CV), combines by
the stated rule, and is saved as one distinct, loadable Model_Version ranked beside its members.
**Validates: Requirements 3.2, 3.4, 3.5, 3.6, 8.3**

### Property 4: stacking is gated and leak-free
A stacking ensemble is refused outside CV_Mode; under CV_Mode its meta-features are OOF (no row's
base prediction comes from a fold model that trained on that row), its base learners are refit on the
full training set, and it is one loadable Model_Version.
**Validates: Requirements 4.2, 4.3, 4.5, 8.4**

### Property 5: fit-cost guard
A stacking ensemble's `base × (K+1) + 1` fits enter the same Fit_Count_Ceiling/budget guards; an
over-ceiling definition is refused with the numbers before anything is archived.
**Validates: Requirements 4.7**

### Property 6: backward compatibility
A training request with no ensemble fields and only existing learners produces byte-identical
artifacts to today; built-in plugins import only base-image libraries.
**Validates: Requirements 7.2, 8.5**

## Error Handling

- An ensemble with <2 base learners, an unknown base/meta learner, or a learner that does not support
  the problem type → `ValidationError` before submission, naming the offender.
- A stacking ensemble in Single_Split_Mode → `ValidationError` "stacking requires CV_Mode" (Req 4.3).
- An installed base/meta learner whose package is absent from `requirements` → `ValidationError`
  naming the missing package (Req 2.3), reusing the requirements check.
- Over the fit-count/budget ceiling → the existing guard's `ValidationError` with the counts (Req 4.7).
- A single ensemble fit failing is recorded for that model and does not abort the others (Req 6.3,
  the inherited per-model isolation).

## Testing Strategy

Offline, `InMemoryObjectStore` + `InlineOrchestrator` + mocked AWS.

- Per new built-in learner: register/available/build/train/predict/importance on a tiny fixture
  (Property 1), and that it imports only scikit-learn (Property 6).
- An installed-learner plugin reports unavailable (not crash) when its library is absent, and the
  service refuses selecting it without the package (Property 2).
- A voting ensemble of two learners: each member fit on training rows only, combined by the rule, one
  distinct loadable Model_Version ranked beside the members (Property 3).
- A stacking ensemble refused in Single_Split_Mode; under CV_Mode the meta-features are leak-free OOF
  and the base learners are refit on the full training set (Property 4).
- The stacking fit-cost guard refuses an over-ceiling definition with the numbers (Property 5).
- A no-ensemble request is byte-identical to today (Property 6, golden).

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/models/registry.py` | Register the new built-in + installed plugins; resolve an ensemble name to an `EnsemblePlugin` built from the `ModelSpec`-carried spec | The one registration/resolution point; additive |
| `ml_engine/contracts/config.py` | Add `ExperimentConfig.ensembles` (optional, empty-excluded) | Freeze the approved ensembles; absence reproduces today |
| `ml_engine/contracts/model.py` | Add optional `ModelMetadata.ensemble_lineage` (excluded when unset) | Base-version lineage of an ensemble (Req 6.4) |
| `backend/schemas.py`, `backend/services/experiments.py` | Ensemble request fields; validate/guard/freeze/expand | Surface, validate and cost ensembles |
| `mcp_server/server.py`, `backend/api/routes/training.py` | Accept + echo the resolved ensembles | Surface what was frozen (Req 5) |

New modules (`contracts/ensemble.py`, `models/boosting.py`, `models/bagging_trees.py`,
`models/lightgbm_plugin.py`, `models/ensemble.py`) are registered as NEW. Each inherited edit is
additive and recorded in `tests/inherited_manifest.json` with a written justification.

## Phasing (open q 1)

- **Phase 1 (no new dependencies):** C1 built-in learners + C3/C4 voting + C5/C6 for voting. Depends
  on nothing beyond today.
- **Phase 2 (gated):** C4 stacking (depends on cross-validation OOF, now present, and the
  model-version store, now present) + C2 open-catalogue installed learners (depends on the
  stock-image `requirements` install, now present). Both land after phase 1 so the registry churns
  once.

## Open questions (resolved in design)

- **Phasing (Q1):** built-in learners + voting first; stacking + open catalogue second, each gated on
  its dependency.
- **Default meta-learner (Q2):** logistic regression (classification) / ridge (regression),
  user-overridable.
- **Base-learner count (Q3):** min 2; max bounded only by the fit-count/budget guards.
- **HistGradientBoosting categoricals (Q4):** shared dense-numeric path for parity; native
  categorical is a later additive flag.
- **Ensembles in the default set (Q5):** no — only when explicitly defined; the default set stays the
  single learners.
