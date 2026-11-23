"""Deployment settings shared by the Streamlit API client."""

from __future__ import annotations

import os


def deployment_setting(name: str) -> str | None:
    """Read an environment variable or Streamlit Community Cloud secret."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    from streamlit import secrets
    from streamlit.errors import StreamlitSecretNotFoundError

    try:
        secret = secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    if not isinstance(secret, str) or not secret.strip():
        return None
    return secret.strip()
