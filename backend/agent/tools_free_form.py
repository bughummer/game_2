"""Free-form experiment tools (design C5 catalogue; Req 3.1, 3.7, 5.7, 10.5, 10.6).

* ``draft_free_form_script`` (Mutating): ``FreeFormService.draft_script``. Returns the draft id
  and revision.
* ``start_free_form_run`` (Costly, ``always_approve``, ``protected``):
  ``FreeFormService.start_run``. It takes ``draft_id`` and ``revision``, never script text.
  The script bytes are loaded server-side through ``FreeFormService.draft_revision``;
  ``free_form_script_for_approval`` gives the code and its content hash for the approval
  view. The user's approval of that request is the script approval, in every mode including
  Autopilot. ``requirements`` and ``use_spot`` pass through to the service exactly as the
  route passes them.
* ``get_free_form_run``, ``list_free_form_runs`` (Read): ``run_status`` and ``list_runs``.
* ``compare_free_form`` (Read, ``verbatim``): ``FreeFormService.compare``, with its
  ``comparable`` flags.

Metric labels (``self_reported`` or ``engine``) pass through unchanged. Storage URIs are removed
from run results (any key ending in ``_uri``) so the model never sees one.
"""

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.agent.tools import (
    AwaitedJobSpec,
    ToolContext,
    ToolFlags,
    ToolOutput,
    ToolRegistry,
)
from backend.errors import ConflictError
from ml_engine.contracts.agent import ToolClass
from ml_engine.contracts.execution import WorkflowKind
from ml_engine.contracts.free_form import (
    DATA_CONTEXT_QUESTIONS,
    DataContextAnswer,
    DataContextAnswers,
    content_hash,
)

FREE_FORM_ESTIMATOR = "free_form"
MAX_DESCRIPTION_CHARS = 4000
MAX_INSTRUCTION_CHARS = 2000
MAX_ANSWER_CHARS = 1000
MAX_REQUIREMENTS = 50
MAX_REQUIREMENT_CHARS = 200

_QUESTIONS = {q.question_id: q.question for q in DATA_CONTEXT_QUESTIONS}
QuestionId = Literal[
    "label_timing",
    "entity_repeats",
    "post_outcome_columns",
    "time_ordered",
    "label_defining_columns",
]
if set(get_args(QuestionId)) != set(_QUESTIONS):  # pragma: no cover - programming error
    raise RuntimeError("QuestionId is out of step with DATA_CONTEXT_QUESTIONS")

DraftId = Annotated[str, StringConstraints(pattern=r"^draft-[A-Za-z0-9_-]{1,64}$")]
RunId = Annotated[str, StringConstraints(pattern=r"^ff-[A-Za-z0-9_-]{1,64}$")]
Requirement = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_REQUIREMENT_CHARS)
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentArgs(_Args):
    experiment_id: str = Field(min_length=1, max_length=200)


class DataContextAnswerArg(_Args):
    question_id: QuestionId
    status: Literal["answered", "dont_know", "skipped"]
    answer: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ANSWER_CHARS),
        ]
        | None
    ) = None


class DraftScriptArgs(ExperimentArgs):
    dataset_id: str = Field(min_length=1, max_length=200, description="A dataset id (ds_...).")
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_DESCRIPTION_CHARS),
    ] = Field(description="What the script should do, in the user's words.")
    data_context: list[DataContextAnswerArg] = Field(
        min_length=1,
        max_length=len(DATA_CONTEXT_QUESTIONS),
        description="The user's answers to the data-context questions, one per question.",
    )
    draft_id: DraftId | None = Field(
        default=None, description="Revise this draft; omit to start a new one."
    )
    instruction: (
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True, min_length=1, max_length=MAX_INSTRUCTION_CHARS
            ),
        ]
        | None
    ) = Field(default=None, description="The user's revise instruction for this revision.")


class StartRunArgs(ExperimentArgs):
    draft_id: DraftId
    revision: int = Field(ge=1, le=10_000)
    requirements: list[Requirement] = Field(
        default_factory=list,
        max_length=MAX_REQUIREMENTS,
        description="pip requirement lines, for example 'statsmodels==0.14.2'.",
    )
    use_spot: bool = False


class RunArgs(ExperimentArgs):
    run_id: RunId | None = Field(default=None, description="Omit for the latest run.")


# --- helpers -------------------------------------------------------------------------------


def strip_uris(value: Any) -> Any:
    """A copy of ``value`` without any dict key ending in ``_uri``. Other values are unchanged."""
    if isinstance(value, dict):
        return {k: strip_uris(v) for k, v in value.items() if not str(k).endswith("_uri")}
    if isinstance(value, list):
        return [strip_uris(v) for v in value]
    return value


def _stored_revision(services: Any, args: StartRunArgs):
    draft, revision = services.free_form.draft_revision(
        args.experiment_id, args.draft_id, args.revision
    )
    if content_hash(revision.content) != revision.content_hash:
        raise ConflictError(
            "The stored draft revision does not match its content hash.",
            {"reason": "CONTENT_HASH_MISMATCH", "draft_id": args.draft_id},
        )
    return draft, revision


def free_form_script_for_approval(ctx: ToolContext, args: StartRunArgs) -> dict[str, Any]:
    """The code and content hash a ``start_free_form_run`` approval must show (Req 3.7)."""
    _draft, revision = _stored_revision(ctx.services, args)
    return {
        "draft_id": args.draft_id,
        "revision": revision.revision,
        "content": revision.content,
        "content_hash": revision.content_hash,
        "assumptions": list(revision.assumptions),
    }


def start_run_protected(_ctx: ToolContext, _args: StartRunArgs) -> bool:
    """A free-form run is always protected: it runs code the platform does not control."""
    return True


def free_form_job_extractor(data: dict[str, Any]) -> list[AwaitedJobSpec]:
    execution_id = data.get("execution_id")
    experiment_id = data.get("experiment_id")
    if not isinstance(execution_id, str) or not isinstance(experiment_id, str):
        return []
    return [
        AwaitedJobSpec(
            execution_id=execution_id,
            experiment_id=experiment_id,
            kind=WorkflowKind.FREE_FORM.value,
        )
    ]


# --- handlers ------------------------------------------------------------------------------


def draft_free_form_script(ctx: ToolContext, args: DraftScriptArgs) -> ToolOutput:
    answers = DataContextAnswers(
        answers=[
            DataContextAnswer(
                question_id=a.question_id,
                question=_QUESTIONS[a.question_id],
                status=a.status,
                answer=a.answer,
            )
            for a in args.data_context
        ]
    )
    # The shell has replaced the dataset id with its URI.
    result = ctx.services.free_form.draft_script(
        args.experiment_id,
        dataset_uri=args.dataset_id,
        description=args.description,
        data_context=answers,
        draft_id=args.draft_id,
        instruction=args.instruction,
    )
    return ToolOutput(
        data=strip_uris(result),
        audit_detail={
            "draft_id": result["draft_id"],
            "revision": str(result["revision"]["revision"]),
        },
    )


def start_free_form_run(ctx: ToolContext, args: StartRunArgs) -> ToolOutput:
    draft, revision = _stored_revision(ctx.services, args)
    status = ctx.services.free_form.start_run(
        args.experiment_id,
        script=revision.content,
        dataset_uri=draft.dataset_uri,
        requirements=list(args.requirements),
        use_spot=args.use_spot,
        data_context=draft.data_context,
        user=ctx.principal.user,
    )
    data = {
        **strip_uris(status),
        "experiment_id": args.experiment_id,
        "draft_id": args.draft_id,
        "revision": revision.revision,
    }
    return ToolOutput(
        data=data,
        audit_detail={
            "run_id": status["run_id"],
            "content_hash": status["content_hash"],
            "dataset_uri": draft.dataset_uri,
        },
    )


def get_free_form_run(ctx: ToolContext, args: RunArgs) -> ToolOutput:
    status = ctx.services.free_form.run_status(args.experiment_id, args.run_id)
    return ToolOutput(data={**strip_uris(status), "experiment_id": args.experiment_id})


def list_free_form_runs(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    runs = ctx.services.free_form.list_runs(args.experiment_id)
    return ToolOutput(data={"experiment_id": args.experiment_id, "runs": strip_uris(runs)})


def compare_free_form(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    result = ctx.services.free_form.compare(
        args.experiment_id, model_versions=ctx.services.model_versions
    )
    return ToolOutput(data=strip_uris(result))


# --- registration --------------------------------------------------------------------------

_QUESTION_LIST = "; ".join(f"{q.question_id}: {q.question}" for q in DATA_CONTEXT_QUESTIONS)


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        "draft_free_form_script",
        classes=[ToolClass.MUTATING],
        description=(
            "Draft (or revise) a free-form Python script for work the managed pipeline does not "
            "cover, such as time series or segmentation. Ask the user these data-context "
            "questions first and pass each answer, or status dont_know or skipped: "
            f"{_QUESTION_LIST}. Returns the draft id and the revision with its code and content "
            "hash. Nothing runs until the user approves start_free_form_run."
        ),
        input_model=DraftScriptArgs,
        handler=draft_free_form_script,
        flags=ToolFlags(
            experiment_arg="experiment_id",
            dataset_args=("dataset_id",),
            audit_action="draft_free_form_script",
        ),
    )
    registry.register(
        "start_free_form_run",
        classes=[ToolClass.COSTLY],
        description=(
            "Run one stored draft revision as a free-form job on the draft's dataset. Always "
            "needs the user's approval, which shows the exact code and its content hash. "
            "Optional pip requirements and use_spot. Metrics come back labelled self_reported "
            "or engine; only engine metrics compare with managed models."
        ),
        input_model=StartRunArgs,
        handler=start_free_form_run,
        flags=ToolFlags(
            always_approve=True,
            experiment_arg="experiment_id",
            estimator=FREE_FORM_ESTIMATOR,
            audit_action="start_free_form",
        ),
        protected=start_run_protected,
        job_extractor=free_form_job_extractor,
        approval_view=free_form_script_for_approval,
    )
    registry.register(
        "get_free_form_run",
        classes=[ToolClass.READ],
        description=(
            "Get one free-form run (the latest unless run_id is given): status, content hash, "
            "leakage screen and labelled metrics."
        ),
        input_model=RunArgs,
        handler=get_free_form_run,
        flags=ToolFlags(experiment_arg="experiment_id"),
    )
    registry.register(
        "list_free_form_runs",
        classes=[ToolClass.READ],
        description="List the experiment's free-form runs, newest first, with labelled metrics.",
        input_model=ExperimentArgs,
        handler=list_free_form_runs,
        flags=ToolFlags(experiment_arg="experiment_id"),
    )
    registry.register(
        "compare_free_form",
        classes=[ToolClass.READ],
        description=(
            "List free-form and managed results together with their metric labels and a "
            "comparable flag. Never rank entries marked comparable false."
        ),
        input_model=ExperimentArgs,
        handler=compare_free_form,
        flags=ToolFlags(experiment_arg="experiment_id", verbatim=True, advisory=True),
    )
