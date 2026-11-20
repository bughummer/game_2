"""Spot checkpointing for the training stage — now inert (stock-image migration, decision B4).

Spot checkpoint/resume is dropped platform-wide: every model is restart-on-interruption and no
job sets a ``CheckpointConfig`` (requirement 5.2). ``RESUMABLE_MODELS`` is empty, so there is
nothing to checkpoint.

The module and its public symbols are kept so the training job's import
(``jobs/training/main.py`` imports ``checkpointing_factory``) still resolves and the change is
minimal and reversible. ``checkpointing_factory`` now returns a no-op factory: the plain plugin,
unchanged, which reads and writes no checkpoint. The CV fold-level per-fold persistence in
``jobs/training/cv.py`` is independent of this module and is unaffected (requirement 5.4).
"""

import logging
from collections.abc import Callable
from pathlib import Path

from ml_engine.contracts.execution import RESUMABLE_MODELS
from ml_engine.models import ModelPlugin, get_plugin

LOGGER = logging.getLogger("ml_factory.jobs.training.checkpointing")


def is_resumable(model_name: str) -> bool:
    """No model resumes from a Spot checkpoint any more (RESUMABLE_MODELS is empty)."""
    return model_name in RESUMABLE_MODELS


def checkpointing_factory(directory: Path) -> Callable[[str], ModelPlugin]:  # noqa: ARG001
    """A no-op ``plugin_factory`` for ``run_training``.

    Checkpoint/resume is dropped (decision B4): every model is restart-on-interruption, so the
    factory returns the plain plugin unchanged — it writes and reads no checkpoint, whatever the
    ``directory`` argument is (kept for signature compatibility with the inherited entrypoint).
    """

    def factory(model_name: str) -> ModelPlugin:
        LOGGER.info("%s is restart-on-interruption; no checkpoints are written", model_name)
        return get_plugin(model_name)

    return factory
