"""boto3 client construction — the one place the control plane builds AWS clients.

Ported from old_repo (``backend/aws/clients.py``) minus static-key support: credentials come
only from the standard chain (on Fargate, the task role) or an assumed role. With
``aws_role_arn`` set, the base identity is used only for ``sts:AssumeRole`` and credentials are
refreshed automatically before they expire, so a long-running process keeps working.
"""

import logging
from typing import Any

from backend.config import Settings

LOGGER = logging.getLogger("ml_factory.aws")


def build_session(settings: Settings) -> Any:
    import boto3

    base_kwargs = {"profile_name": settings.aws_profile} if settings.aws_profile else {}
    if not settings.aws_role_arn:
        return boto3.session.Session(region_name=settings.aws_region, **base_kwargs)
    return _assumed_role_session(settings.aws_role_arn, settings.aws_region, settings.aws_profile)


def _assumed_role_session(role_arn: str, region: str, profile: str | None) -> Any:
    """A session holding ``role_arn``'s credentials, re-assumed before expiry.

    This is the provider botocore builds for a ``role_arn`` profile, constructed by hand so the
    role comes from configuration. Assigning ``_credentials`` is what botocore does internally.
    """
    import boto3
    from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials
    from botocore.session import Session as BotocoreSession

    base = BotocoreSession()
    base.set_config_variable("region", region)
    if profile:
        base.set_config_variable("profile", profile)
    fetcher = AssumeRoleCredentialFetcher(
        client_creator=base.create_client,
        source_credentials=base.get_credentials(),
        role_arn=role_arn,
        extra_args={"RoleSessionName": "ml-factory-control-plane"},
    )
    assumed = BotocoreSession()
    assumed.set_config_variable("region", region)
    assumed._credentials = DeferredRefreshableCredentials(
        method="assume-role", refresh_using=fetcher.fetch_credentials
    )
    return boto3.Session(botocore_session=assumed)


def botocore_config(settings: Settings, **overrides: Any) -> Any:
    from botocore.config import Config

    proxies = {"https": settings.https_proxy} if settings.https_proxy else None
    if proxies:
        LOGGER.info("Using the configured HTTPS proxy")  # the URL may carry credentials
    return Config(
        region_name=settings.aws_region,
        retries={"max_attempts": 5, "mode": "standard"},
        proxies=proxies,
        user_agent_extra="ml-factory/2.0",
        **overrides,
    )


def build_client(service: str, settings: Settings, **overrides: Any) -> Any:
    """A retry-configured, proxy-aware, credential-aware boto3 client."""
    return build_session(settings).client(service, config=botocore_config(settings, **overrides))
