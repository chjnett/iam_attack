"""Deterministic IAM episodes for the one-month FlowGate feasibility pilot.

The corpus is intentionally small and auditable: four matched pairs for each
of three escalation families.  A pair has the same IAM state and API sequence;
only coarse operational facts differ.  Intent labels are returned through a
separate trusted-plane object and are never embedded in :class:`Episode`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any

from .models import (
    Episode,
    EpisodeLabel,
    Event,
    FAMILY_PASSROLE_COMPUTE,
    FAMILY_POLICY_ATTACHMENT,
    FAMILY_ROLE_TRUST,
    IAMState,
    ManagedPolicy,
    PrincipalConfig,
    ResourceConfig,
    RoleConfig,
    SPLIT_BLIND,
    SPLIT_PROMPT_DEV,
    Statement,
)


_SENSITIVE_ACTION = "secretsmanager:GetSecretValue"


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _operational_facts(pair_index: int, *, malicious: bool) -> dict[str, bool]:
    """Return observable facts, not an intent label.

    The patterns deliberately overlap: no single flag is a perfect classifier.
    This keeps the pilot focused on permission evidence plus operational
    context instead of turning it into a one-bit lookup task.
    """

    suspicious_side = (
        {
            "change_ticket_present": False,
            "approved_actor_match": False,
            "maintenance_window_match": False,
            "rollback_observed": False,
            "break_glass": False,
        },
        {
            "change_ticket_present": True,
            "approved_actor_match": False,
            "maintenance_window_match": True,
            "rollback_observed": False,
            "break_glass": False,
        },
        {
            "change_ticket_present": False,
            "approved_actor_match": True,
            "maintenance_window_match": False,
            "rollback_observed": False,
            "break_glass": False,
        },
        {
            "change_ticket_present": True,
            "approved_actor_match": False,
            "maintenance_window_match": False,
            "rollback_observed": True,
            "break_glass": False,
        },
    )
    authorized_side = (
        {
            "change_ticket_present": True,
            "approved_actor_match": True,
            "maintenance_window_match": True,
            "rollback_observed": False,
            "break_glass": False,
        },
        {
            "change_ticket_present": True,
            "approved_actor_match": True,
            "maintenance_window_match": False,
            "rollback_observed": True,
            "break_glass": True,
        },
        {
            "change_ticket_present": False,
            "approved_actor_match": True,
            "maintenance_window_match": False,
            "rollback_observed": False,
            "break_glass": True,
        },
        {
            "change_ticket_present": True,
            "approved_actor_match": True,
            "maintenance_window_match": True,
            "rollback_observed": True,
            "break_glass": False,
        },
    )
    source = suspicious_side if malicious else authorized_side
    return dict(source[pair_index])


def _boundary(
    *, sensitive_action: str, sensitive_resource_id: str, blocks_sensitive: bool
) -> tuple[Statement, ...] | None:
    if not blocks_sensitive:
        return None
    return (
        Statement(
            effect="deny",
            actions=(sensitive_action,),
            resources=(sensitive_resource_id,),
        ),
        Statement(effect="allow", actions=("*",), resources=("*",)),
    )


def _event(
    *,
    episode_id: str,
    ordinal: int,
    base_time: datetime,
    action: str,
    actor: str,
    target: str = "",
    resource: str = "",
    outcome: str = "success",
    details: dict[str, Any] | None = None,
) -> Event:
    return Event.create(
        event_id=f"{episode_id}-E{ordinal:02d}",
        timestamp=_iso(base_time + timedelta(minutes=ordinal)),
        action=action,
        actor=actor,
        target=target,
        resource=resource,
        outcome=outcome,
        details=details,
    )


def _policy_attachment_episode(
    *,
    episode_id: str,
    pair_index: int,
    split: str,
    malicious: bool,
    base_time: datetime,
) -> Episode:
    suffix = f"1{pair_index + 1:02d}"
    actor_id = f"P{suffix}"
    role_id = f"R{suffix}"
    policy_id = f"Y{suffix}"
    resource_id = f"X{suffix}"
    blocks_sensitive = pair_index == 1
    state = IAMState(
        as_of=_iso(base_time),
        principals=(
            PrincipalConfig(
                principal_id=actor_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=("iam:AttachRolePolicy", "sts:AssumeRole"),
                        resources=(role_id,),
                    ),
                ),
            ),
        ),
        roles=(
            RoleConfig(
                role_id=role_id,
                statements=(),
                trusted_principals=(actor_id,),
                boundary=_boundary(
                    sensitive_action=_SENSITIVE_ACTION,
                    sensitive_resource_id=resource_id,
                    blocks_sensitive=blocks_sensitive,
                ),
            ),
        ),
        managed_policies=(
            ManagedPolicy(
                policy_id=policy_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=(_SENSITIVE_ACTION,),
                        resources=(resource_id,),
                    ),
                ),
            ),
        ),
        resources=(
            ResourceConfig(
                resource_id=resource_id,
                resource_type="secret",
                sensitive=True,
            ),
        ),
        complete=pair_index != 3,
    )
    facts = _operational_facts(pair_index, malicious=malicious)
    events = [
        _event(
            episode_id=episode_id,
            ordinal=1,
            base_time=base_time,
            action="iam:AttachRolePolicy",
            actor=actor_id,
            target=role_id,
            resource=role_id,
            details={**facts, "policy_id": policy_id},
        ),
        _event(
            episode_id=episode_id,
            ordinal=2,
            base_time=base_time,
            action="sts:AssumeRole",
            actor=actor_id,
            target=role_id,
            resource=role_id,
        ),
    ]
    if pair_index in {0, 1}:
        events.append(
            _event(
                episode_id=episode_id,
                ordinal=3,
                base_time=base_time,
                action=_SENSITIVE_ACTION,
                actor=f"role-session:{role_id}",
                resource=resource_id,
                outcome="success" if pair_index == 0 else "failure",
            )
        )
    return Episode(
        episode_id=episode_id,
        family=FAMILY_POLICY_ATTACHMENT,
        split=split,
        actor_id=actor_id,
        target_role_id=role_id,
        sensitive_action=_SENSITIVE_ACTION,
        sensitive_resource_id=resource_id,
        initial_state=state,
        events=tuple(events),
    )


def _role_trust_episode(
    *,
    episode_id: str,
    pair_index: int,
    split: str,
    malicious: bool,
    base_time: datetime,
) -> Episode:
    suffix = f"2{pair_index + 1:02d}"
    actor_id = f"P{suffix}"
    role_id = f"R{suffix}"
    resource_id = f"X{suffix}"
    blocks_sensitive = pair_index == 1
    state = IAMState(
        as_of=_iso(base_time),
        principals=(
            PrincipalConfig(
                principal_id=actor_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=("iam:UpdateAssumeRolePolicy", "sts:AssumeRole"),
                        resources=(role_id,),
                    ),
                ),
            ),
        ),
        roles=(
            RoleConfig(
                role_id=role_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=(_SENSITIVE_ACTION,),
                        resources=(resource_id,),
                    ),
                ),
                trusted_principals=("P000",),
                boundary=_boundary(
                    sensitive_action=_SENSITIVE_ACTION,
                    sensitive_resource_id=resource_id,
                    blocks_sensitive=blocks_sensitive,
                ),
            ),
        ),
        managed_policies=(),
        resources=(
            ResourceConfig(
                resource_id=resource_id,
                resource_type="secret",
                sensitive=True,
            ),
        ),
        complete=pair_index != 3,
    )
    facts = _operational_facts(pair_index, malicious=malicious)
    events = [
        _event(
            episode_id=episode_id,
            ordinal=1,
            base_time=base_time,
            action="iam:UpdateAssumeRolePolicy",
            actor=actor_id,
            target=role_id,
            resource=role_id,
            details={
                **facts,
                "added_principal": actor_id,
                "trust_condition": "true",
            },
        ),
        _event(
            episode_id=episode_id,
            ordinal=2,
            base_time=base_time,
            action="sts:AssumeRole",
            actor=actor_id,
            target=role_id,
            resource=role_id,
        ),
    ]
    if pair_index in {0, 1}:
        events.append(
            _event(
                episode_id=episode_id,
                ordinal=3,
                base_time=base_time,
                action=_SENSITIVE_ACTION,
                actor=f"role-session:{role_id}",
                resource=resource_id,
                outcome="success" if pair_index == 0 else "failure",
            )
        )
    return Episode(
        episode_id=episode_id,
        family=FAMILY_ROLE_TRUST,
        split=split,
        actor_id=actor_id,
        target_role_id=role_id,
        sensitive_action=_SENSITIVE_ACTION,
        sensitive_resource_id=resource_id,
        initial_state=state,
        events=tuple(events),
    )


def _passrole_compute_episode(
    *,
    episode_id: str,
    pair_index: int,
    split: str,
    malicious: bool,
    base_time: datetime,
) -> Episode:
    suffix = f"3{pair_index + 1:02d}"
    actor_id = f"P{suffix}"
    role_id = f"R{suffix}"
    compute_id = f"C{suffix}"
    resource_id = f"X{suffix}"
    compute_action = (
        "lambda:CreateFunction" if pair_index % 2 == 0 else "ec2:RunInstances"
    )
    service_principal = (
        "lambda.amazonaws.com"
        if compute_action.startswith("lambda:")
        else "ec2.amazonaws.com"
    )
    blocks_sensitive = pair_index == 1
    state = IAMState(
        as_of=_iso(base_time),
        principals=(
            PrincipalConfig(
                principal_id=actor_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=("iam:PassRole",),
                        resources=(role_id,),
                    ),
                    Statement(
                        effect="allow",
                        actions=(
                            (compute_action, "lambda:InvokeFunction")
                            if compute_action == "lambda:CreateFunction"
                            else (compute_action,)
                        ),
                        resources=(compute_id,),
                    ),
                ),
            ),
        ),
        roles=(
            RoleConfig(
                role_id=role_id,
                statements=(
                    Statement(
                        effect="allow",
                        actions=(_SENSITIVE_ACTION,),
                        resources=(resource_id,),
                    ),
                ),
                trusted_principals=(service_principal,),
                boundary=_boundary(
                    sensitive_action=_SENSITIVE_ACTION,
                    sensitive_resource_id=resource_id,
                    blocks_sensitive=blocks_sensitive,
                ),
            ),
        ),
        managed_policies=(),
        resources=(
            ResourceConfig(
                resource_id=resource_id,
                resource_type="secret",
                sensitive=True,
            ),
        ),
        complete=pair_index != 3,
    )
    facts = _operational_facts(pair_index, malicious=malicious)
    events = [
        _event(
            episode_id=episode_id,
            ordinal=1,
            base_time=base_time,
            action=compute_action,
            actor=actor_id,
            target=compute_id,
            resource=compute_id,
            details={
                **facts,
                "execution_role": role_id,
                "code_controlled": True,
            },
        )
    ]
    if compute_action == "lambda:CreateFunction":
        events.append(
            _event(
                episode_id=episode_id,
                ordinal=2,
                base_time=base_time,
                action="lambda:InvokeFunction",
                actor=actor_id,
                target=compute_id,
                resource=compute_id,
            )
        )
    if pair_index in {0, 1}:
        events.append(
            _event(
                episode_id=episode_id,
                ordinal=3 if compute_action == "lambda:CreateFunction" else 2,
                base_time=base_time,
                action=_SENSITIVE_ACTION,
                actor=f"service-session:{role_id}",
                resource=resource_id,
                outcome="success" if pair_index == 0 else "failure",
            )
        )
    return Episode(
        episode_id=episode_id,
        family=FAMILY_PASSROLE_COMPUTE,
        split=split,
        actor_id=actor_id,
        target_role_id=role_id,
        sensitive_action=_SENSITIVE_ACTION,
        sensitive_resource_id=resource_id,
        initial_state=state,
        events=tuple(events),
    )


_BUILDERS = {
    FAMILY_POLICY_ATTACHMENT: _policy_attachment_episode,
    FAMILY_ROLE_TRUST: _role_trust_episode,
    FAMILY_PASSROLE_COMPUTE: _passrole_compute_episode,
}


def _opaque_episode_id(slot: int) -> str:
    """Return a deterministic ID whose visible ordering does not encode intent."""

    token = hashlib.sha256(f"flowgate-pilot-v1:{slot}".encode("utf-8")).hexdigest()[:10]
    return f"FG-{token}"


def build_fixture_corpus() -> tuple[tuple[Episode, ...], tuple[EpisodeLabel, ...]]:
    """Return 24 episodes plus 24 labels held on a separate trusted plane."""

    episodes: list[Episode] = []
    labels: list[EpisodeLabel] = []
    episode_number = 1
    prompt_pair_by_family = {
        FAMILY_POLICY_ATTACHMENT: 0,
        FAMILY_ROLE_TRUST: 1,
        FAMILY_PASSROLE_COMPUTE: 3,
    }
    for family_index, family in enumerate(_BUILDERS):
        builder = _BUILDERS[family]
        for pair_index in range(4):
            split = (
                SPLIT_PROMPT_DEV
                if pair_index == prompt_pair_by_family[family]
                else SPLIT_BLIND
            )
            base_time = datetime(
                2026,
                1,
                2 + family_index * 4 + pair_index,
                12,
                tzinfo=timezone.utc,
            )
            for malicious in (True, False):
                episode_id = _opaque_episode_id(episode_number)
                episode = builder(
                    episode_id=episode_id,
                    pair_index=pair_index,
                    split=split,
                    malicious=malicious,
                    base_time=base_time,
                )
                episodes.append(episode)
                labels.append(
                    EpisodeLabel(
                        episode_id=episode_id,
                        malicious=malicious,
                        family=family,
                        split=split,
                    )
                )
                episode_number += 1
    return tuple(episodes), tuple(labels)


__all__ = ["build_fixture_corpus"]
