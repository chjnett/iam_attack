from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PREFIXES = {
    "principal": "P",
    "role": "R",
    "resource": "X",
    "policy": "Y",
    "event": "E",
    "generic": "O",
}


@dataclass
class OpaqueMapper:
    """Create study-local opaque IDs without exposing reversible IAM names.

    The mapping is retained only by the trusted laptop process.  Exported jobs
    contain values such as P001 and R002, never account IDs or ARNs.
    """

    mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    def map(self, value: str | None, kind: str = "generic") -> str | None:
        if value is None:
            return None
        table = self.mappings.setdefault(kind, {})
        if value not in table:
            prefix = PREFIXES.get(kind, PREFIXES["generic"])
            table[value] = f"{prefix}{len(table) + 1:03d}"
        return table[value]

    def map_list(self, values: list[str], kind: str) -> list[str]:
        return [mapped for value in values if (mapped := self.map(value, kind)) is not None]

    def export_private_map(self) -> dict[str, dict[str, str]]:
        """Return a copy for laptop-only archival; never attach this to GPU jobs."""

        return {kind: dict(values) for kind, values in self.mappings.items()}


def sanitize_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep only coarse, predeclared operational signals.

    Free-form ticket text, usernames, IPs, tags, and comments are intentionally
    excluded from the inference contract.
    """

    allowed = {
        "change_ticket_present",
        "approved_actor_match",
        "maintenance_window_match",
        "rollback_observed",
        "break_glass",
    }
    return {key: bool(context[key]) for key in sorted(allowed & context.keys())}

