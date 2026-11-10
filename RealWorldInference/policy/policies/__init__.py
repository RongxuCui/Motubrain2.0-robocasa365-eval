"""Policy loader — dynamically imports and instantiates policy classes."""

import importlib
import logging

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


def _find_policy_class(module, policy_type: str):
    """Find the Policy class in a module by naming convention."""
    # Try: FALCONPolicy, FalconPolicy, falconPolicy, then any *Policy
    candidates = [
        f"{policy_type.upper()}Policy",
        f"{policy_type.capitalize()}Policy",
        "".join(w.capitalize() for w in policy_type.split("_")) + "Policy",
    ]
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BasePolicy) and obj is not BasePolicy:
            return obj

    raise AttributeError(f"No BasePolicy subclass found in policies.{policy_type}.policy")


def load_policy(policy_type: str, policy_config: dict) -> BasePolicy:
    """Load a policy by type name.

    Args:
        policy_type: subdirectory name under policies/ (e.g. "falcon", "dummy")
        policy_config: full policy config dict, passed as **kwargs to the constructor
    """
    module = importlib.import_module(f"policy.policies.{policy_type}.policy")
    cls = _find_policy_class(module, policy_type)
    logger.info(f"Loading policy: {policy_type} ({cls.__name__})")
    return cls(**policy_config)
