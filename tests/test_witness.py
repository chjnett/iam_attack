from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flowgate.authz import evaluate_prefix
from flowgate.fixtures import build_fixture_corpus
from flowgate.models import (
    CapabilityState,
    FAMILIES,
    FAMILY_PASSROLE_COMPUTE,
    SPLIT_BLIND,
    SPLIT_PROMPT_DEV,
)
from flowgate.witness import build_witness, verify_witness_digest


_OPERATIONAL_KEYS = {
    "change_ticket_present",
    "approved_actor_match",
    "maintenance_window_match",
    "rollback_observed",
    "break_glass",
}
_FORBIDDEN_LABEL_KEYS = {
    "attack",
    "expected_verdict",
    "ground_truth",
    "label",
    "malicious",
}


def _all_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            found.add(str(key))
            found.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_all_keys(nested))
    return found


def _matched_shape(payload: dict[str, object]) -> dict[str, object]:
    clone = json.loads(json.dumps(payload, sort_keys=True))
    clone.pop("episode_id")
    for event in clone["events"]:
        event.pop("event_id")
        for key in _OPERATIONAL_KEYS:
            event["details"].pop(key, None)
    return clone


class FixtureCorpusTests(unittest.TestCase):
    def test_exact_size_balance_families_and_splits(self) -> None:
        episodes, labels = build_fixture_corpus()
        self.assertEqual(len(episodes), 24)
        self.assertEqual(len(labels), 24)
        self.assertEqual(set(item.family for item in episodes), set(FAMILIES))
        self.assertEqual(Counter(item.family for item in episodes), {f: 8 for f in FAMILIES})
        self.assertEqual(sum(item.malicious for item in labels), 12)
        self.assertEqual(Counter(item.split for item in episodes)[SPLIT_PROMPT_DEV], 6)
        self.assertEqual(Counter(item.split for item in episodes)[SPLIT_BLIND], 18)
        self.assertEqual(
            {item.episode_id for item in episodes},
            {item.episode_id for item in labels},
        )

    def test_labels_are_separate_and_episode_payloads_are_label_blind(self) -> None:
        episodes, labels = build_fixture_corpus()
        self.assertTrue(any(label.malicious for label in labels))
        for episode in episodes:
            self.assertFalse(hasattr(episode, "malicious"))
            present = _all_keys(episode.to_dict())
            self.assertFalse(present & _FORBIDDEN_LABEL_KEYS)

    def test_each_attack_has_a_structurally_matched_benign_episode(self) -> None:
        episodes, labels = build_fixture_corpus()
        label_by_id = {item.episode_id: item for item in labels}
        for offset in range(0, len(episodes), 2):
            first, second = episodes[offset : offset + 2]
            self.assertTrue(label_by_id[first.episode_id].malicious)
            self.assertFalse(label_by_id[second.episode_id].malicious)
            self.assertEqual(_matched_shape(first.to_dict()), _matched_shape(second.to_dict()))

    def test_fixture_construction_is_deterministic(self) -> None:
        self.assertEqual(build_fixture_corpus(), build_fixture_corpus())

    def test_visible_episode_id_does_not_encode_label_by_sequence_or_parity(self) -> None:
        episodes, labels = build_fixture_corpus()
        label_by_id = {item.episode_id: item.malicious for item in labels}
        self.assertFalse(all(item.episode_id == f"FG-{index:03d}" for index, item in enumerate(episodes, 1)))
        parity_predictions = {
            episode.episode_id: bool(int(episode.episode_id.split("-", 1)[1], 16) & 1)
            for episode in episodes
        }
        accuracy = sum(
            parity_predictions[episode_id] == malicious
            for episode_id, malicious in label_by_id.items()
        ) / len(labels)
        self.assertNotIn(accuracy, {0.0, 1.0})


class WitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes, cls.labels = build_fixture_corpus()
        cls.by_id = {episode.episode_id: episode for episode in cls.episodes}

    def test_final_provenance_distribution(self) -> None:
        results = [evaluate_prefix(episode) for episode in self.episodes]
        self.assertEqual(
            Counter(result.capability_state for result in results),
            {
                CapabilityState.VERIFIED: 12,
                CapabilityState.POSSIBLE: 6,
                CapabilityState.UNKNOWN: 6,
            },
        )
        gains = Counter(result.capability_gain for result in results)
        self.assertEqual(gains[True], 12)
        self.assertEqual(gains[False], 6)
        self.assertEqual(gains[None], 6)

    def test_prefix_is_strict_and_future_events_do_not_leak(self) -> None:
        episode = self.episodes[0]
        prefix = build_witness(episode, prefix_len=1)
        full = build_witness(episode)
        payload = prefix.to_dict()
        serialized = json.dumps(payload, sort_keys=True)
        self.assertEqual(payload["prefix_len"], 1)
        self.assertEqual(payload["cutoff"], episode.events[0].timestamp)
        self.assertEqual(
            [item["event_id"] for item in payload["observed_events"]],
            [episode.events[0].event_id],
        )
        for future in episode.events[1:]:
            self.assertNotIn(future.event_id, serialized)
            self.assertNotIn(future.timestamp, serialized)
        self.assertNotEqual(prefix.witness_digest, full.witness_digest)

    def test_witness_has_stable_keys_no_labels_and_valid_digest(self) -> None:
        witness = build_witness(self.episodes[0])
        payload = witness.to_dict()
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "episode_id",
                "family",
                "split",
                "prefix_len",
                "cutoff",
                "capability_state",
                "capability_gain",
                "provenance",
                "observed_events",
                "candidate_path",
                "unknown_preconditions",
                "feature_summary",
                "witness_digest",
            },
        )
        self.assertFalse(_all_keys(payload) & _FORBIDDEN_LABEL_KEYS)
        self.assertEqual(len(payload["witness_digest"]), 64)
        self.assertEqual(payload["witness_digest"], payload["witness_digest"].lower())
        self.assertTrue(verify_witness_digest(witness))
        tampered = dict(payload)
        tampered["cutoff"] = "2099-01-01T00:00:00Z"
        self.assertFalse(verify_witness_digest(tampered))

    def test_builder_cannot_receive_a_label(self) -> None:
        parameters = set(inspect.signature(build_witness).parameters)
        self.assertEqual(parameters, {"episode", "prefix_len"})
        before = build_witness(self.episodes[0]).witness_digest
        changed_label = replace(self.labels[0], malicious=not self.labels[0].malicious)
        self.assertNotEqual(changed_label.malicious, self.labels[0].malicious)
        after = build_witness(self.episodes[0]).witness_digest
        self.assertEqual(before, after)

    def test_permission_boundary_blocks_gain_with_verified_provenance(self) -> None:
        for episode in (self.episodes[2], self.episodes[10], self.episodes[18]):
            result = evaluate_prefix(episode)
            self.assertEqual(result.capability_state, CapabilityState.VERIFIED)
            self.assertFalse(result.capability_gain)
            self.assertTrue(result.candidate_path)
            self.assertFalse(result.unknown_preconditions)

    def test_unknown_snapshot_fails_closed(self) -> None:
        for episode in (self.episodes[6], self.episodes[14], self.episodes[22]):
            witness = build_witness(episode)
            self.assertEqual(witness.capability_state, CapabilityState.UNKNOWN)
            self.assertIsNone(witness.capability_gain)
            self.assertIn("initial_state_incomplete", witness.unknown_preconditions)

    def test_passrole_is_inferred_not_fabricated_as_cloudtrail_event(self) -> None:
        episode = self.episodes[16]
        witness = build_witness(episode)
        self.assertEqual(episode.family, FAMILY_PASSROLE_COMPUTE)
        self.assertIn("iam:PassRole", witness.candidate_path)
        self.assertIn("lambda:InvokeFunction", witness.candidate_path)
        self.assertNotIn(
            "iam:PassRole", {event.action for event in witness.observed_events}
        )

    def test_lambda_sensitive_use_requires_an_observed_invocation(self) -> None:
        episode = self.episodes[16]
        without_invoke = replace(
            episode,
            events=tuple(
                event
                for event in episode.events
                if event.action != "lambda:InvokeFunction"
            ),
        )
        result = evaluate_prefix(without_invoke)
        self.assertEqual(result.capability_state, CapabilityState.POSSIBLE)
        self.assertTrue(result.capability_gain)
        self.assertFalse(result.sensitive_action_observed)

    def test_sensitive_use_must_be_bound_to_expected_role_session(self) -> None:
        episode = self.episodes[0]
        unrelated_use = replace(episode.events[-1], actor="P999")
        altered = replace(episode, events=episode.events[:-1] + (unrelated_use,))
        result = evaluate_prefix(altered)
        self.assertEqual(result.capability_state, CapabilityState.POSSIBLE)
        self.assertTrue(result.capability_gain)
        self.assertFalse(result.sensitive_action_observed)

    def test_causal_mutation_must_be_performed_by_episode_actor(self) -> None:
        for episode in (self.episodes[0], self.episodes[8], self.episodes[16]):
            foreign = replace(episode.events[0], actor="PZZZ")
            altered = replace(episode, events=(foreign,) + episode.events[1:])
            result = evaluate_prefix(altered)
            self.assertEqual(result.capability_state, CapabilityState.VERIFIED)
            self.assertFalse(result.capability_gain)
            self.assertFalse(result.sensitive_action_observed)

    def test_passrole_sensitive_use_must_match_created_compute_target(self) -> None:
        episode = self.episodes[16]
        invoke_index = next(
            index
            for index, event in enumerate(episode.events)
            if event.action == "lambda:InvokeFunction"
        )
        changed = list(episode.events)
        changed[invoke_index] = replace(changed[invoke_index], target="C-OTHER")
        altered = replace(episode, events=tuple(changed))
        result = evaluate_prefix(altered)
        self.assertEqual(result.capability_state, CapabilityState.POSSIBLE)
        self.assertTrue(result.capability_gain)
        self.assertFalse(result.sensitive_action_observed)

    def test_redundant_policy_attachment_is_not_a_capability_gain(self) -> None:
        episode = self.episodes[0]
        policy_id = str(episode.events[0].detail("policy_id"))
        role = next(
            role for role in episode.initial_state.roles if role.role_id == episode.target_role_id
        )
        changed_role = replace(
            role,
            attached_policy_ids=tuple(sorted({*role.attached_policy_ids, policy_id})),
        )
        state = replace(
            episode.initial_state,
            roles=tuple(
                changed_role if item.role_id == changed_role.role_id else item
                for item in episode.initial_state.roles
            ),
        )
        result = evaluate_prefix(replace(episode, initial_state=state))
        self.assertEqual(result.capability_state, CapabilityState.VERIFIED)
        self.assertFalse(result.capability_gain)

    def test_redundant_trust_update_is_not_a_capability_gain(self) -> None:
        episode = self.episodes[8]
        role = next(
            role for role in episode.initial_state.roles if role.role_id == episode.target_role_id
        )
        changed_role = replace(
            role,
            trusted_principals=tuple(
                sorted({*role.trusted_principals, episode.actor_id})
            ),
            trust_condition="true",
        )
        state = replace(
            episode.initial_state,
            roles=tuple(
                changed_role if item.role_id == changed_role.role_id else item
                for item in episode.initial_state.roles
            ),
        )
        result = evaluate_prefix(replace(episode, initial_state=state))
        self.assertEqual(result.capability_state, CapabilityState.VERIFIED)
        self.assertFalse(result.capability_gain)

    def test_invalid_prefix_is_rejected(self) -> None:
        episode = self.episodes[0]
        with self.assertRaises(ValueError):
            build_witness(episode, prefix_len=-1)
        with self.assertRaises(ValueError):
            build_witness(episode, prefix_len=len(episode.events) + 1)


if __name__ == "__main__":
    unittest.main()
