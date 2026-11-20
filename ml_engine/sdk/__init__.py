"""The Platform_SDK a free-form script may import (free-form-experiments design C2).

A thin, versioned façade over the engine's own pure code — the same splitters and metric functions
the managed pipeline uses, so a number produced through the SDK equals the managed pipeline's for
the same inputs (Req 4.3). Importing it is opt-in; a script that uses it gets engine-computed,
fold-safe numbers that the platform labels ``engine`` rather than ``self_reported``.

Pure ``ml_engine``: no IO, no AWS, no control plane. It imports only base-image libraries, so the
offline suite and a no-requirements run never need anything extra.

    from ml_engine.sdk import splits, metrics, results, SDK_VERSION
"""

from ml_engine.sdk import metrics, results, splits

SDK_VERSION = "1.0"

__all__ = ["SDK_VERSION", "metrics", "results", "splits"]
