"""Scoped, fail-closed authorization evaluator for the FlowGate pilot.

This is not a complete AWS IAM implementation.  It deliberately supports the
three fixture families and a closed subset of IAM semantics: identity policies,
managed policy attachment, permissions boundaries, session policies, SCP deny
statements, role trust, and explicit deny precedence.  Unknown condition or
state inputs propagate to ``CapabilityState.UNKNOWN`` instead of being treated
as an allow.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Iterable

from .models import (
    AuthzResult,
    CapabilityState,
    Decision,
    Episode,
    Event,
    FAMILY_PASSROLE_COMPUTE,
    FAMILY_POLICY_ATTACHMENT,
    FAMILY_ROLE_TRUST,
    IAMState,
    PrincipalConfig,
    RoleConfig,
    Statement,
)


def _matches(statement: Statement, action: str, resource: str) -> bool:
    action_lc = action.lower()
    action_match = any(
        fnmatchcase(action_lc, pattern.lower()) for pattern in statement.actions
    )
    resource_match = any(
        fnmatchcase(resource, pattern) for pattern in statement.resources
    )
    return action_match and resource_match


def _statement_decision(
    statements: Iterable[Statement], action: str, resource: str
) -> Decision:
    """Evaluate a policy list using explicit-deny precedence.

    A matching statement with an unknown condition makes the result unknown
    unless a known explicit deny already fixes the result.  A known allow does
    not override a potentially applicable unknown deny.
    """

    known_allow = False
    unknown_allow = False
    unknown_deny = False
    for statement in statements:
        if not _matches(statement, action, resource):
            continue
        if statement.condition == "false":
            continue
        if statement.condition == "unknown":
            if statement.effect == "deny":
                unknown_deny = True
            else:
                unknown_allow = True
            continue
        if statement.effect == "deny":
            return Decision.DENY
        known_allow = True
    if unknown_deny:
        return Decision.UNKNOWN
    if known_allow:
        return Decision.ALLOW
    if unknown_allow:
        return Decision.UNKNOWN
    return Decision.DENY


def _intersect(base: Decision, limiter: Decision) -> Decision:
    if base is Decision.DENY or limiter is Decision.DENY:
        return Decision.DENY
    if base is Decision.UNKNOWN or limiter is Decision.UNKNOWN:
        return Decision.UNKNOWN
    return Decision.ALLOW


@dataclass
class _RuntimeState:
    source: IAMState
    principals: dict[str, PrincipalConfig]
    roles: dict[str, RoleConfig]
    managed_policies: dict[str, tuple[Statement, ...]]
    role_attachments: dict[str, set[str]]
    trusted_principals: dict[str, set[str]]
    trust_conditions: dict[str, str]
    assumed_roles: set[tuple[str, str]]
    delegated_roles: set[tuple[str, str, str]]
    unknown_preconditions: set[str]
    provenance: list[str]

    @classmethod
    def from_state(cls, state: IAMState) -> "_RuntimeState":
        return cls(
            source=state,
            principals={item.principal_id: item for item in state.principals},
            roles={item.role_id: item for item in state.roles},
            managed_policies={
                item.policy_id: item.statements for item in state.managed_policies
            },
            role_attachments={
                item.role_id: set(item.attached_policy_ids) for item in state.roles
            },
            trusted_principals={
                item.role_id: set(item.trusted_principals) for item in state.roles
            },
            trust_conditions={
                item.role_id: item.trust_condition for item in state.roles
            },
            assumed_roles=set(),
            delegated_roles=set(),
            unknown_preconditions=set(),
            provenance=[
                "snapshot:complete" if state.complete else "snapshot:incomplete"
            ],
        )


def _attached_statements(
    runtime: _RuntimeState, attached_policy_ids: Iterable[str]
) -> tuple[Statement, ...]:
    statements: list[Statement] = []
    for policy_id in sorted(attached_policy_ids):
        policy = runtime.managed_policies.get(policy_id)
        if policy is None:
            runtime.unknown_preconditions.add(f"managed_policy_missing:{policy_id}")
            continue
        statements.extend(policy)
    return tuple(statements)


def _apply_scp(
    runtime: _RuntimeState, decision: Decision, action: str, resource: str
) -> Decision:
    """Apply the fixture's SCP deny-list semantics.

    Fixtures intentionally contain deny statements only.  Encountering an SCP
    allow statement is outside this pilot's semantics and therefore unknown.
    """

    applicable = [
        statement
        for statement in runtime.source.scp_statements
        if _matches(statement, action, resource)
    ]
    if any(statement.effect == "allow" for statement in applicable):
        runtime.unknown_preconditions.add("scp_allow_list_semantics_unsupported")
        return Decision.UNKNOWN
    for statement in applicable:
        if statement.condition == "true":
            return Decision.DENY
        if statement.condition == "unknown":
            runtime.unknown_preconditions.add("scp_condition_context_missing")
            return Decision.UNKNOWN
    return decision


def _principal_decision(
    runtime: _RuntimeState, principal_id: str, action: str, resource: str
) -> Decision:
    principal = runtime.principals.get(principal_id)
    if principal is None:
        runtime.unknown_preconditions.add(f"principal_missing:{principal_id}")
        return Decision.UNKNOWN
    statements = principal.statements + _attached_statements(
        runtime, principal.attached_policy_ids
    )
    decision = _statement_decision(statements, action, resource)
    if principal.boundary is not None:
        decision = _intersect(
            decision, _statement_decision(principal.boundary, action, resource)
        )
    if principal.session_policy is not None:
        decision = _intersect(
            decision,
            _statement_decision(principal.session_policy, action, resource),
        )
    return _apply_scp(runtime, decision, action, resource)


def _role_decision(
    runtime: _RuntimeState, role_id: str, action: str, resource: str
) -> Decision:
    role = runtime.roles.get(role_id)
    if role is None:
        runtime.unknown_preconditions.add(f"role_missing:{role_id}")
        return Decision.UNKNOWN
    statements = role.statements + _attached_statements(
        runtime, runtime.role_attachments.get(role_id, set())
    )
    decision = _statement_decision(statements, action, resource)
    if role.boundary is not None:
        decision = _intersect(
            decision, _statement_decision(role.boundary, action, resource)
        )
    return _apply_scp(runtime, decision, action, resource)


def _trust_decision(
    runtime: _RuntimeState, principal_id: str, role_id: str
) -> Decision:
    role = runtime.roles.get(role_id)
    if role is None:
        runtime.unknown_preconditions.add(f"role_missing:{role_id}")
        return Decision.UNKNOWN
    trusted = runtime.trusted_principals.get(role_id, set())
    if principal_id not in trusted and "*" not in trusted:
        return Decision.DENY
    condition = runtime.trust_conditions.get(role_id, "unknown")
    if condition == "unknown":
        runtime.unknown_preconditions.add(f"trust_condition_missing:{role_id}")
        return Decision.UNKNOWN
    if condition == "false":
        return Decision.DENY
    return Decision.ALLOW


def _event_resource(event: Event) -> str:
    return event.resource or event.target or "*"


def _record_event(runtime: _RuntimeState, event: Event, episode: Episode) -> None:
    runtime.provenance.append(f"event:{event.event_id}:{event.outcome}")
    if event.outcome == "failure":
        return
    if event.outcome == "unknown":
        runtime.unknown_preconditions.add(f"event_outcome_missing:{event.event_id}")
        return

    # The pilot's causal claim is explicitly about the untrusted principal's
    # own actions.  Do not combine another principal's mutation with this
    # episode actor's permissions to manufacture an escalation path.
    actor_id = episode.actor_id

    if event.action == "iam:AttachRolePolicy":
        if event.actor != actor_id or event.target != episode.target_role_id:
            return
        policy_id = str(event.detail("policy_id", ""))
        if not policy_id:
            runtime.unknown_preconditions.add(
                f"attached_policy_id_missing:{event.event_id}"
            )
            return
        runtime.role_attachments.setdefault(event.target, set()).add(policy_id)
        runtime.provenance.append(
            f"state:policy_attached:{event.target}:{policy_id}"
        )
        return

    if event.action == "iam:UpdateAssumeRolePolicy":
        if event.actor != actor_id or event.target != episode.target_role_id:
            return
        added_principal = str(event.detail("added_principal", actor_id))
        runtime.trusted_principals.setdefault(event.target, set()).add(
            added_principal
        )
        condition = str(event.detail("trust_condition", "true")).lower()
        if condition not in {"true", "false", "unknown"}:
            condition = "unknown"
        runtime.trust_conditions[event.target] = condition
        runtime.provenance.append(
            f"state:trust_updated:{event.target}:{added_principal}"
        )
        return

    if event.action == "sts:AssumeRole":
        if event.actor != actor_id or event.target != episode.target_role_id:
            return
        runtime.assumed_roles.add((event.actor, event.target))
        runtime.provenance.append(f"observed:assume_role:{event.target}")
        return

    if event.action in {"lambda:CreateFunction", "ec2:RunInstances"}:
        if event.actor != actor_id:
            return
        execution_role = str(event.detail("execution_role", ""))
        code_controlled = event.detail("code_controlled", None)
        if not execution_role:
            runtime.unknown_preconditions.add(
                f"execution_role_missing:{event.event_id}"
            )
            return
        if execution_role != episode.target_role_id:
            return
        if code_controlled is True:
            runtime.delegated_roles.add(
                (actor_id, execution_role, event.target or event.resource)
            )
            runtime.provenance.append(
                f"state:compute_delegation:{execution_role}:{event.target}"
            )
        elif code_controlled is None:
            runtime.unknown_preconditions.add(
                f"code_control_context_missing:{event.event_id}"
            )


def _all_known_allow(decisions: Iterable[tuple[str, Decision]]) -> tuple[bool | None, int]:
    satisfied = 0
    saw_unknown = False
    for _, decision in decisions:
        if decision is Decision.DENY:
            return False, satisfied
        if decision is Decision.UNKNOWN:
            saw_unknown = True
        else:
            satisfied += 1
    if saw_unknown:
        return None, satisfied
    return True, satisfied


def _baseline_role_capability(episode: Episode, runtime: _RuntimeState) -> bool | None:
    """Return whether the actor already had the target capability at snapshot time."""

    decisions = (
        (
            "assume_permission",
            _principal_decision(
                runtime, episode.actor_id, "sts:AssumeRole", episode.target_role_id
            ),
        ),
        (
            "role_trust",
            _trust_decision(runtime, episode.actor_id, episode.target_role_id),
        ),
        (
            "sensitive_permission",
            _role_decision(
                runtime,
                episode.target_role_id,
                episode.sensitive_action,
                episode.sensitive_resource_id,
            ),
        ),
    )
    reachable, _ = _all_known_allow(decisions)
    runtime.provenance.append(
        "baseline_capability:"
        + ("unknown" if reachable is None else "present" if reachable else "absent")
    )
    return reachable


def _sensitive_observation(episode: Episode, events: tuple[Event, ...]) -> bool:
    expected_actor = (
        f"service-session:{episode.target_role_id}"
        if episode.family == FAMILY_PASSROLE_COMPUTE
        else f"role-session:{episode.target_role_id}"
    )
    for index, event in enumerate(events):
        if not (
            event.action == episode.sensitive_action
            and event.resource == episode.sensitive_resource_id
            and event.actor == expected_actor
            and event.outcome == "success"
        ):
            continue
        earlier = events[:index]
        if episode.family in {FAMILY_POLICY_ATTACHMENT, FAMILY_ROLE_TRUST}:
            mutation_action = (
                "iam:AttachRolePolicy"
                if episode.family == FAMILY_POLICY_ATTACHMENT
                else "iam:UpdateAssumeRolePolicy"
            )
            mutation_index = next(
                (
                    offset
                    for offset, prior in enumerate(earlier)
                    if prior.action == mutation_action
                    and prior.actor == episode.actor_id
                    and prior.target == episode.target_role_id
                    and prior.outcome == "success"
                ),
                None,
            )
            if mutation_index is None:
                continue
            if any(
                prior.action == "sts:AssumeRole"
                and prior.actor == episode.actor_id
                and prior.target == episode.target_role_id
                and prior.outcome == "success"
                for prior in earlier[mutation_index + 1 :]
            ):
                return True
            continue

        compute_index = next(
            (
                offset
                for offset, prior in enumerate(earlier)
                if prior.action in {"lambda:CreateFunction", "ec2:RunInstances"}
                and prior.actor == episode.actor_id
                and prior.outcome == "success"
                and prior.detail("execution_role", "") == episode.target_role_id
                and prior.detail("code_controlled", None) is True
            ),
            None,
        )
        if compute_index is None:
            continue
        compute_event = earlier[compute_index]
        if compute_event.action == "ec2:RunInstances":
            return True
        if any(
            prior.action == "lambda:InvokeFunction"
            and prior.actor == episode.actor_id
            and prior.target == compute_event.target
            and prior.outcome == "success"
            for prior in earlier[compute_index + 1 :]
        ):
            return True
    return False


def _policy_mutation_path(
    episode: Episode, runtime: _RuntimeState, events: tuple[Event, ...]
) -> tuple[bool | None, tuple[str, ...], int, int]:
    baseline = _RuntimeState.from_state(episode.initial_state)
    baseline_reachable = _baseline_role_capability(episode, baseline)
    runtime.unknown_preconditions.update(baseline.unknown_preconditions)
    runtime.provenance.extend(baseline.provenance[1:])
    attach_seen = any(
        event.action == "iam:AttachRolePolicy"
        and event.actor == episode.actor_id
        and event.target == episode.target_role_id
        and event.outcome != "failure"
        for event in events
    )
    mutation = _principal_decision(
        runtime,
        episode.actor_id,
        "iam:AttachRolePolicy",
        episode.target_role_id,
    )
    assume = _principal_decision(
        runtime,
        episode.actor_id,
        "sts:AssumeRole",
        episode.target_role_id,
    )
    trust = _trust_decision(runtime, episode.actor_id, episode.target_role_id)
    sensitive = _role_decision(
        runtime,
        episode.target_role_id,
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    decisions = (
        ("mutation_permission", mutation),
        ("assume_permission", assume),
        ("role_trust", trust),
        ("sensitive_permission", sensitive),
    )
    if not attach_seen:
        # No temporal capability change has occurred at this prefix.
        return False, (), 4, sum(d is Decision.ALLOW for _, d in decisions)
    reachable, satisfied = _all_known_allow(decisions)
    if baseline_reachable is None:
        reachable = None
    elif baseline_reachable:
        # A redundant/no-op mutation is not a before→after privilege gain.
        reachable = False
    path = (
        episode.actor_id,
        "iam:AttachRolePolicy",
        episode.target_role_id,
        "sts:AssumeRole",
        f"role-session:{episode.target_role_id}",
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    return reachable, path, 4, satisfied


def _role_trust_path(
    episode: Episode, runtime: _RuntimeState, events: tuple[Event, ...]
) -> tuple[bool | None, tuple[str, ...], int, int]:
    baseline = _RuntimeState.from_state(episode.initial_state)
    baseline_reachable = _baseline_role_capability(episode, baseline)
    runtime.unknown_preconditions.update(baseline.unknown_preconditions)
    runtime.provenance.extend(baseline.provenance[1:])
    update_seen = any(
        event.action == "iam:UpdateAssumeRolePolicy"
        and event.actor == episode.actor_id
        and event.target == episode.target_role_id
        and event.outcome != "failure"
        for event in events
    )
    mutation = _principal_decision(
        runtime,
        episode.actor_id,
        "iam:UpdateAssumeRolePolicy",
        episode.target_role_id,
    )
    assume = _principal_decision(
        runtime,
        episode.actor_id,
        "sts:AssumeRole",
        episode.target_role_id,
    )
    trust = _trust_decision(runtime, episode.actor_id, episode.target_role_id)
    sensitive = _role_decision(
        runtime,
        episode.target_role_id,
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    decisions = (
        ("trust_mutation_permission", mutation),
        ("assume_permission", assume),
        ("role_trust", trust),
        ("sensitive_permission", sensitive),
    )
    if not update_seen:
        return False, (), 4, sum(d is Decision.ALLOW for _, d in decisions)
    reachable, satisfied = _all_known_allow(decisions)
    if baseline_reachable is None:
        reachable = None
    elif baseline_reachable:
        reachable = False
    path = (
        episode.actor_id,
        "iam:UpdateAssumeRolePolicy",
        episode.target_role_id,
        "sts:AssumeRole",
        f"role-session:{episode.target_role_id}",
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    return reachable, path, 4, satisfied


def _passrole_path(
    episode: Episode, runtime: _RuntimeState, events: tuple[Event, ...]
) -> tuple[bool | None, tuple[str, ...], int, int]:
    compute_event = next(
        (
            event
            for event in events
            if event.action in {"lambda:CreateFunction", "ec2:RunInstances"}
            and event.actor == episode.actor_id
            and event.outcome != "failure"
            and event.detail("execution_role", "") == episode.target_role_id
            and bool(event.target or event.resource)
        ),
        None,
    )
    if compute_event is None:
        return False, (), 4, 0
    service_principal = (
        "lambda.amazonaws.com"
        if compute_event.action.startswith("lambda:")
        else "ec2.amazonaws.com"
    )
    pass_role = _principal_decision(
        runtime,
        episode.actor_id,
        "iam:PassRole",
        episode.target_role_id,
    )
    create_compute = _principal_decision(
        runtime,
        episode.actor_id,
        compute_event.action,
        _event_resource(compute_event),
    )
    service_trust = _trust_decision(
        runtime, service_principal, episode.target_role_id
    )
    sensitive = _role_decision(
        runtime,
        episode.target_role_id,
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    decisions: tuple[tuple[str, Decision], ...] = (
        ("passrole_permission", pass_role),
        ("compute_permission", create_compute),
    )
    if compute_event.action == "lambda:CreateFunction":
        invoke = _principal_decision(
            runtime,
            episode.actor_id,
            "lambda:InvokeFunction",
            _event_resource(compute_event),
        )
        decisions += (("invoke_permission", invoke),)
    decisions += (
        ("service_trust", service_trust),
        ("sensitive_permission", sensitive),
    )
    reachable, satisfied = _all_known_allow(decisions)
    if compute_event.detail("code_controlled", None) is None:
        runtime.unknown_preconditions.add(
            f"code_control_context_missing:{compute_event.event_id}"
        )
        reachable = None
    elif compute_event.detail("code_controlled", None) is not True:
        reachable = False
    path_prefix = (
        episode.actor_id,
        "iam:PassRole",
        episode.target_role_id,
        compute_event.action,
        compute_event.target or compute_event.resource,
    )
    if compute_event.action == "lambda:CreateFunction":
        path_prefix += (
            "lambda:InvokeFunction",
            compute_event.target or compute_event.resource,
        )
    path = path_prefix + (
        f"service-session:{episode.target_role_id}",
        episode.sensitive_action,
        episode.sensitive_resource_id,
    )
    return reachable, path, len(decisions), satisfied


def evaluate_prefix(episode: Episode, prefix_len: int | None = None) -> AuthzResult:
    """Evaluate one episode using only events visible in the requested prefix."""

    events = episode.prefix(prefix_len)
    runtime = _RuntimeState.from_state(episode.initial_state)
    for event in events:
        _record_event(runtime, event, episode)

    if not episode.initial_state.complete:
        runtime.unknown_preconditions.add("initial_state_incomplete")

    if episode.family == FAMILY_POLICY_ATTACHMENT:
        reachable, path, required, satisfied = _policy_mutation_path(
            episode, runtime, events
        )
    elif episode.family == FAMILY_ROLE_TRUST:
        reachable, path, required, satisfied = _role_trust_path(
            episode, runtime, events
        )
    elif episode.family == FAMILY_PASSROLE_COMPUTE:
        reachable, path, required, satisfied = _passrole_path(
            episode, runtime, events
        )
    else:  # Episode validates this, but fail closed if a foreign object arrives.
        runtime.unknown_preconditions.add(f"unsupported_family:{episode.family}")
        reachable, path, required, satisfied = None, (), 1, 0

    sensitive_observed = _sensitive_observation(episode, events)
    if runtime.unknown_preconditions:
        capability_state = CapabilityState.UNKNOWN
        capability_gain: bool | None = None
    elif reachable is None:
        capability_state = CapabilityState.UNKNOWN
        capability_gain = None
    elif reachable is False:
        capability_state = CapabilityState.VERIFIED
        capability_gain = False
    elif sensitive_observed:
        capability_state = CapabilityState.VERIFIED
        capability_gain = True
        matching_event = next(
            event
            for event in events
            if event.action == episode.sensitive_action
            and event.resource == episode.sensitive_resource_id
            and event.actor
            == (
                f"service-session:{episode.target_role_id}"
                if episode.family == FAMILY_PASSROLE_COMPUTE
                else f"role-session:{episode.target_role_id}"
            )
            and event.outcome == "success"
        )
        runtime.provenance.append(f"observed:sensitive_use:{matching_event.event_id}")
    else:
        capability_state = CapabilityState.POSSIBLE
        capability_gain = True

    runtime.provenance.append(f"capability_state:{capability_state.value}")
    return AuthzResult(
        capability_state=capability_state,
        capability_gain=capability_gain,
        candidate_path=tuple(path),
        unknown_preconditions=tuple(sorted(runtime.unknown_preconditions)),
        provenance=tuple(runtime.provenance),
        sensitive_action_observed=sensitive_observed,
        required_preconditions=required,
        satisfied_preconditions=satisfied,
    )


__all__ = ["evaluate_prefix"]
