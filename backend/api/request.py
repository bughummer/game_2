"""Reading a request body and query parameters for the REST API (control-ui-api design C3).

The handlers are plain Starlette coroutines, so they parse their own inputs. ``json_body`` tolerates
an empty body (a POST with no payload is ``{}``) and refuses a non-object JSON document, turning a
malformed body into a ``ValidationError`` the ``safe`` wrapper maps to 400 rather than a 500.
"""

import json
from typing import Any

from starlette.requests import Request

from backend.errors import ValidationError


async def json_body(request: Request) -> dict[str, Any]:
    """The request's JSON object, or ``{}`` when the body is empty."""
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError("The request body is not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValidationError("The request body must be a JSON object.")
    return parsed


def query_int(request: Request, name: str, default: int) -> int:
    """A positive-ish integer query parameter, falling back to ``default`` when absent/malformed."""
    value = request.query_params.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValidationError(f"Query parameter '{name}' must be an integer.") from error


def query_flag(request: Request, name: str) -> bool:
    """A truthy query flag: present and not one of the explicit false spellings."""
    value = request.query_params.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}
