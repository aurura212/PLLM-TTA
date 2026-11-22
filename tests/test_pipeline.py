"""Regression checks for the comparison protocol and offline experiment replay."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np

from tpop2.data import load_tasks, normalize_question, probe_questions, profiles
from tpop2.decode import LocalDecoder, PrefixProbe, choose_token
from tpop2.experiment import build_report, read_jsonl, run_fixed, run_generate, training_pair
from tpop2.feedback import FeedbackClient, FeedbackError
from tpop2.reward import choice_difference, eligible_constraints, extract_features, fit_reward, loss_and_gradient

ROOT = Path(__file__).resolve().parent.parent
TASKS = load_tasks(ROOT / "data" / "controlled_tasks.jsonl")


class ProtocolTests(unittest.TestCase):
    def test_controlled_splits_and_variants(self):
        self.assertEqual(Counter(task.split for task in TASKS), {"train": 16, "dev": 8, "test": 16})
        self.assertEqual(len({normalize_question(task.question) for task in TASKS}), 40)
        self.assertEqual(len(profiles()), 8)
        for task in TASKS:
            for lead in (0, 1):
                for example in (0, 1):
                    text = task.render(lead, example)
                    self.assertEqual(extract_features(text), (lead, example))
                    self.assertIn(task.answer, text)
                    self.assertIn(task.explanation, text)

    def test_probe_prompts_are_disjoint_and_repeatable(self):
        excluded = TASKS[0].question
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            samples = [{"question": f"Explain concept {i}."} for i in range(105)]
            samples.extend([{"question": excluded.upper()}, {"question": "  EXPLAIN CONCEPT 0.  "}])
            path.write_text(json.dumps(samples), encoding="utf-8")
            train, valid = probe_questions(path, TASKS)
            self.assertEqual((len(train), len(valid)), (72, 24))
            self.assertEqual((train, valid), probe_questions(path, TASKS))
            keys = {normalize_question(question) for question in train + valid}
            self.assertEqual(len(keys), 96)
            self.assertNotIn(normalize_question(excluded), keys)

    def test_winner_label_and_zero_gradient_on_missing_attribute(self):
        for profile in profiles():
            a, b, missing = training_pair(profile, 17)
            diff = choice_difference(a, b, "A")
            np.testing.assert_array_equal(choice_difference(a, b, "B"), -diff)
            self.assertEqual(diff[("lead", "example").index(missing)], 0)
            _, gradient = loss_and_gradient(np.zeros(2), diff[None, :], np.ones(1), 1.0)
            self.assertGreater(float((-gradient) @ diff), 0)
        tape = [{"a_features": [1, 0], "b_features": [0, 0], "choice": "A", "comments": {}}]
        self.assertGreater(fit_reward(tape, "choice")[0], 0)
        self.assertEqual(fit_reward(tape, "choice")[1], 0)

    def test_future_constraints_and_missing_feedback(self):
        raw = "Next time add an example. For this answer, lead with the conclusion."
        claims = [
            {"attribute": "example", "direction": 1, "scope": "future", "evidence": "Next time add an example."},
            {"attribute": "lead", "direction": 1, "scope": "current", "evidence": "lead with the conclusion"},
            {"attribute": "lead", "direction": -1, "scope": "future", "evidence": "invented quote"},
        ]
        self.assertEqual(eligible_constraints(raw, claims), [claims[0]])
        tape = [{"a_features": [1, 0], "b_features": [0, 0], "choice": "A",
                 "comments": {"critique": {"raw": raw, "claims": claims}}}]
        np.testing.assert_array_equal(fit_reward(tape, "choice"), fit_reward(tape, "reason"))
        self.assertGreater(fit_reward(tape, "critique")[1], 0)
        tape[0]["comments"]["critique"]["claims"] = claims[1:]
        np.testing.assert_array_equal(fit_reward(tape, "choice"), fit_reward(tape, "critique"))

    def test_prefix_probe_persistence_and_reward_scoring(self):
        x = [[0, 0], [1, 0], [0, 1], [1, 1]]
        probe = PrefixProbe.fit(x, x)
        expected = probe.predict(x)
        self.assertGreater(expected[3, 0], expected[0, 0])
        self.assertEqual(choose_token([5, 7], [0.6, 0.4]), 5)
        self.assertEqual(choose_token([5, 7], [0.6, 0.4], [[0, 0], [1, 1]], [1, 1]), 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.npz"
            probe.save(path, "test-model")
            np.testing.assert_array_equal(expected, PrefixProbe.load(path, "test-model").predict(x))
            with self.assertRaisesRegex(ValueError, "different model"):
                PrefixProbe.load(path, "other-model")

    def test_local_decode_eos_length_and_common_context_budget(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for the fixed-answer stage")

        class TokenModel:
            def __init__(self, sequence):
                self.sequence = iter(sequence)
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                logits = torch.zeros((1, 1, 4))
                logits[0, 0, next(self.sequence)] = 1
                return SimpleNamespace(logits=logits, past_key_values=None)

        decoder = LocalDecoder.__new__(LocalDecoder)
        decoder.torch = torch
        decoder.logit_kwargs = {}
        decoder.eos_ids = {3}
        decoder.context_size = 8
        decoder._prompt = lambda question, history: torch.tensor([[0, 0]])
        decoder._text = lambda tokens: tuple(token for token in tokens if token != 3)
        decoder.model = TokenModel([1, 3])
        self.assertEqual(decoder.generate("test", max_new_tokens=5), (1,))
        self.assertEqual(decoder.model.calls, 2)
        decoder.model = TokenModel([1, 2])
        self.assertEqual(decoder.generate("test", max_new_tokens=2), (1, 2))
        self.assertEqual(decoder.model.calls, 2)
        with self.assertRaisesRegex(ValueError, "context window"):
            decoder.generate("test", max_new_tokens=7)


def _preference(profile, features):
    values = dict(zip(("lead", "example"), features))
    return sum((2 if name == profile["priority"] else 1) * (profile[name] == value)
               for name, value in values.items())


class OfflineExperimentTests(unittest.TestCase):
    def test_simulator_to_report_replays_and_keeps_roles_separate(self):
        calls = Counter()

        def transport(payload):
            model = payload["model"]
            system, user = payload["messages"]
            data = json.loads(user["content"])
            self.assertNotIn("private_context", user["content"])
            a, b = extract_features(data["A"]), extract_features(data["B"])
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            if "remark" in data:
                calls["parse"] += 1
                self.assertEqual(set(data), {"question", "A", "B", "choice", "remark"})
                attr, value = data["remark"].removeprefix("Next time: ").split("=")
                return json.dumps({"items": [{"attribute": attr, "direction": 2 * int(value) - 1,
                                              "scope": "future", "evidence": data["remark"]}]})
            if "preferences" in data:
                calls["judge"] += 1
                self.assertEqual(model, "judge-model")
                self.assertEqual(set(data), {"preferences", "question", "reference", "A", "B"})
                self.assertNotIn("id", data["preferences"])
                self.assertIn("anonymous answers", system["content"])
                left, right = _preference(data["preferences"], a), _preference(data["preferences"], b)
                return json.dumps({"winner": "A" if left > right else "B" if left < right else "tie",
                                   "correct_a": True, "correct_b": True,
                                   "lead_a": bool(a[0]), "lead_b": bool(b[0]),
                                   "example_a": bool(a[1]), "example_b": bool(b[1])})
            self.assertEqual(model, "user-model")
            profile = data["profile"]
            self.assertNotIn("id", profile)
            if "follow_up" in data:
                calls["comments"] += 1
                attr = profile["priority"]
                if data["follow_up"].startswith("For future answers"):
                    attr = "example" if attr == "lead" else "lead"
                return json.dumps({"comment": f"Next time: {attr}={profile[attr]}"})
            calls["choice"] += 1
            return json.dumps({"choice": "A" if _preference(profile, a) >= _preference(profile, b) else "B"})

        questions = {task.question: task for task in TASKS if task.split == "test"}

        class TestDecoder:
            model_id = "offline-test-decoder"

            def generate(self, question, *, weights, probe, history, **kwargs):
                if history:
                    self_parent.assertIsNone(weights)
                    self_parent.assertIsNone(probe)
                    records = json.loads(history)
                    self_parent.assertEqual(len(records), 16)
                    for row in records:
                        self_parent.assertEqual(set(row), {"question", "A", "B", "selected", "comment"})
                    return questions[question].render(0, 1)
                self_parent.assertIsNotNone(probe)
                return questions[question].render(*(int(weight > 1e-7) for weight in weights))

        self_parent = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_client = lambda: FeedbackClient(root / "cache", "user-model", "judge-model", transport=transport)
            client = make_client()
            fixed, generation = root / "fixed", root / "generation"
            run_fixed(TASKS, client, fixed, seeds=(17,), profile_limit=1, test_limit=2)
            tape = read_jsonl(fixed / "feedback.jsonl")
            self.assertEqual(len(tape), 16)
            self.assertEqual([row["round"] for row in tape if row["comments"]], [4, 8, 12, 16])
            measures = json.loads((fixed / "fixed_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(measures["comments"]["critique"]["eligible_missing"], 4)
            self.assertEqual(measures["comments"]["reason"]["eligible_missing"], 0)
            main_rows = [row for row in measures["results"] if row["pair"] == "missing_only" and row["round"] == 16]
            self.assertEqual({row["group"]: row["mean"] for row in main_rows},
                             {"choice": 0.5, "reason": 0.5, "critique": 1.0})
            saved = (fixed / "fixed_predictions.jsonl").read_bytes()
            counts = calls.copy()
            run_fixed(TASKS, make_client(), fixed, seeds=(17,), profile_limit=1, test_limit=2)
            self.assertEqual(counts, calls)
            self.assertEqual(saved, (fixed / "fixed_predictions.jsonl").read_bytes())
            run_generate(TASKS, client, TestDecoder(), object(), fixed, generation, max_new_tokens=8)
            self.assertEqual(len(read_jsonl(generation / "generations.jsonl")), 8)
            self.assertEqual(len(read_jsonl(generation / "generation_scores.jsonl")), 6)
            self.assertEqual(calls["judge"], 6)
            for path in (fixed / "fixed_metrics.json", generation / "generation_metrics.json"):
                expected = path.read_bytes()
                path.unlink()
                self.assertEqual(build_report(path.parent), [path])
                self.assertEqual(path.read_bytes(), expected)

    def test_api_retries_and_leaves_invalid_evidence_uncached(self):
        profile = profiles()[0]
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def transport(payload):
                calls.append(payload)
                if len(calls) == 1:
                    raise HTTPError("https://test.invalid", 429, "retry", {}, None)
                if "remark" in json.loads(payload["messages"][1]["content"]):
                    return '{"items":[{"attribute":"lead","direction":1,"scope":"future","evidence":"fabricated"}]}'
                return '{"choice":"A"}'

            client = FeedbackClient(directory, "user-model", "judge-model", transport=transport,
                                    sleep=lambda delay: None)
            self.assertEqual(client.choose(profile, "q", "a", "b", 17, "private_context"), "A")
            for _ in range(2):
                with self.assertRaisesRegex(FeedbackError, "private_context.*evidence"):
                    client.parse_comment("q", "a", "b", "A", "Make it clearer.", "reason", 17, "private_context")
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
