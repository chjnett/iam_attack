"""Core data models for the CPU-side FlowGate pilot.

The module intentionally keeps intent labels separate from episodes.  Episode
objects contain only observable configuration, event, and organization-context
facts.  ``EpisodeLabel`` belongs to the trusted evaluation plane and must never
be serialized into a witness sent to a model worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"

FAMILY_POLICY_ATTACHMENT = "policy_attachment_abuse"
FAMILY_ROLE_TRUST = "role_trust_abuse"
FAMILY_PASSROLE_COMPUTE = "passrole_compute_abuse"
FAMILIES = (
    FAMILY_POLICY_ATTACHMENT,
    FAMILY_ROLE_TRUST,
    FAMILY_PASSROLE_COMPUTE,
)

SPLIT_PROMPT_DEV = "prompt_dev"
SPLIT_BLIND = "blind"
SPLITS = (SPLIT_PROMPT_DEV, SPLIT_BLIND)


class CapabilityState(str, Enum):
    """Evidence status for a capability claim.

    ``verified`` means the evaluator knows all scoped preconditions and has an
    observed successful use of the sensitive capability (or has verified that
    no gain exists at the current prefix). ``possible`` means a fully known
    path is open but sensitive use has not been observed. ``unknown`` means at
    least one required authorization or telemetry precondition is unavailable.
    """

    VERIFIED = "verified"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy of a sanitized value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported sanitized JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Statement:
    """Small, scoped IAM statement used by the pilot evaluator.

    ``condition`` is deliberately closed: fixtures may mark it ``true``,
    ``false``, or ``unknown``.  The pilot never guesses arbitrary AWS
    Condition semantics.
    """

    effect: str
    actions: tuple[str, ...]
    resources: tuple[str, ...] = ("*",)
    condition: str = "true"

    def __post_init__(self) -> None:
        effect = self.effect.lower()
        condition = self.condition.lower()
        if effect not in {"allow", "deny"}:
            raise ValueError(f"unsupported statement effect: {self.effect}")
        if condition not in {"true", "false", "unknown"}:
            raise ValueError(f"unsupported condition state: {self.condition}")
        if not self.actions or not self.resources:
            raise ValueError("statement actions and resources must be non-empty")
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "resources", tuple(self.resources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect,
            "actions": list(self.actions),
            "resources": list(self.resources),
            "condition": self.condition,
        }


@dataclass(frozen=True, slots=True)
class ManagedPolicy:
    policy_id: str
    statements: tuple[Statement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "statements": [statement.to_dict() for statement in self.statements],
        }


@dataclass(frozen=True, slots=True)
class PrincipalConfig:
    principal_id: str
    statements: tuple[Statement, ...]
    attached_policy_ids: tuple[str, ...] = ()
    boundary: tuple[Statement, ...] | None = None
    session_policy: tuple[Statement, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "statements": [statement.to_dict() for statement in self.statements],
            "attached_policy_ids": list(self.attached_policy_ids),
            "boundary": (
                None
                if self.boundary is None
                else [statement.to_dict() for statement in self.boundary]
            ),
            "session_policy": (
                None
                if self.session_policy is None
                else [statement.to_dict() for statement in self.session_policy]
            ),
        }


@dataclass(frozen=True, slots=True)
class RoleConfig:
    role_id: str
    statements: tuple[Statement, ...]
    trusted_principals: tuple[str, ...]
    attached_policy_ids: tuple[str, ...] = ()
    trust_condition: str = "true"
    boundary: tuple[Statement, ...] | None = None

    def __post_init__(self) -> None:
        condition = self.trust_condition.lower()
        if condition not in {"true", "false", "unknown"}:
            raise ValueError(f"unsupported trust condition: {self.trust_condition}")
        object.__setattr__(self, "trust_condition", condition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "statements": [statement.to_dict() for statement in self.statements],
            "trusted_principals": list(self.trusted_principals),
            "attached_policy_ids": list(self.attached_policy_ids),
            "trust_condition": self.trust_condition,
            "boundary": (
                None
                if self.boundary is None
                else [statement.to_dict() for statement in self.boundary]
            ),
        }


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    resource_id: str
    resource_type: str
    sensitive: bool
    resource_policy: tuple[Statement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "sensitive": self.sensitive,
            "resource_policy": [
                statement.to_dict() for statement in self.resource_policy
            ],
        }


@dataclass(frozen=True, slots=True)
class IAMState:
    as_of: str
    principals: tuple[PrincipalConfig, ...]
    roles: tuple[RoleConfig, ...]
    managed_policies: tuple[ManagedPolicy, ...]
    resources: tuple[ResourceConfig, ...]
    scp_statements: tuple[Statement, ...] = ()
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "principals": [principal.to_dict() for principal in self.principals],
            "roles": [role.to_dict() for role in self.roles],
            "managed_policies": [policy.to_dict() for policy in self.managed_policies],
            "resources": [resource.to_dict() for resource in self.resources],
            "scp_statements": [statement.to_dict() for statement in self.scp_statements],
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    timestamp: str
    action: str
    actor: str
    target: str
    resource: str
    outcome: str
    details: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        outcome = self.outcome.lower()
        if outcome not in {"success", "failure", "unknown"}:
            raise ValueError(f"unsupported event outcome: {self.outcome}")
        object.__setattr__(self, "outcome", outcome)
        normalized = tuple(sorted((str(key), value) for key, value in self.details))
        # Validate the values now so worker bundles cannot acquire arbitrary objects.
        _json_safe(dict(normalized))
        object.__setattr__(self, "details", normalized)

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        timestamp: str,
        action: str,
        actor: str,
        target: str = "",
        resource: str = "",
        outcome: str = "success",
        details: Mapping[str, Any] | None = None,
    ) -> "Event":
        return cls(
            event_id=event_id,
            timestamp=timestamp,
            action=action,
            actor=actor,
            target=target,
            resource=resource,
            outcome=outcome,
            details=tuple((details or {}).items()),
        )

    def detail(self, key: str, default: Any = None) -> Any:
        return dict(self.details).get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "target": self.target,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": _json_safe(dict(self.details)),
        }


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    family: str
    split: str
    actor_id: str
    target_role_id: str
    sensitive_action: str
    sensitive_resource_id: str
    initial_state: IAMState
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported family: {self.family}")
        if self.split not in SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError(f"duplicate event ID in {self.episode_id}")
        timestamps = [event.timestamp for event in self.events]
        if timestamps != sorted(timestamps):
            raise ValueError(f"events are not time ordered in {self.episode_id}")

    def prefix(self, prefix_len: int | None = None) -> tuple[Event, ...]:
        if prefix_len is None:
            prefix_len = len(self.events)
        if prefix_len < 0 or prefix_len > len(self.events):
            raise ValueError(
                f"prefix_len must be in [0, {len(self.events)}], got {prefix_len}"
            )
        return self.events[:prefix_len]

    def cutoff(self, prefix_len: int | None = None) -> str:
        observed = self.prefix(prefix_len)
        return observed[-1].timestamp if observed else self.initial_state.as_of

    def to_dict(self) -> dict[str, Any]:
        """Serialize observable episode data only; no intent label is present."""

        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "family": self.family,
            "split": self.split,
            "actor_id": self.actor_id,
            "target_role_id": self.target_role_id,
            "sensitive_action": self.sensitive_action,
            "sensitive_resource_id": self.sensitive_resource_id,
            "initial_state": self.initial_state.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True, slots=True)
class EpisodeLabel:
    """Trusted-plane label; never an input to witness generation."""

    episode_id: str
    malicious: bool
    family: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "malicious": self.malicious,
            "family": self.family,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class AuthzResult:
    capability_state: CapabilityState
    capability_gain: bool | None
    candidate_path: tuple[str, ...]
    unknown_preconditions: tuple[str, ...]
    provenance: tuple[str, ...]
    sensitive_action_observed: bool
    required_preconditions: int
    satisfied_preconditions: int


@dataclass(frozen=True, slots=True)
class Witness:
    episode_id: str
    family: str
    split: str
    prefix_len: int
    cutoff: str
    capability_state: CapabilityState
    capability_gain: bool | None
    provenance: tuple[str, ...]
    observed_events: tuple[Event, ...]
    candidate_path: tuple[str, ...]
    unknown_preconditions: tuple[str, ...]
    feature_summary: tuple[tuple[str, Any], ...]
    witness_digest: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "family": self.family,
            "split": self.split,
            "prefix_len": self.prefix_len,
            "cutoff": self.cutoff,
            "capability_state": self.capability_state.value,
            "capability_gain": self.capability_gain,
            "provenance": list(self.provenance),
            "observed_events": [event.to_dict() for event in self.observed_events],
            "candidate_path": list(self.candidate_path),
            "unknown_preconditions": list(self.unknown_preconditions),
            "feature_summary": _json_safe(dict(self.feature_summary)),
            "witness_digest": self.witness_digest,
        }
