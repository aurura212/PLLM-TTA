"""Offline checks of the shared upstream path and the optional critique term."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# The offline tests drive chat completions through in-memory responses.
if importlib.util.find_spec("openai") is None:
    with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=Mock())}):
        import judger
    sys.modules["judger"] = judger
else:
    import judger

import imple_armo
from winrate_evaluator import WinrateEvaluator


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def collector(contents, *, critiques):
    with patch.object(judger, "OpenAI"):
        instance = judger.PreferenceCollector("unused", "simulated-user", sleep_time=0, max_retries=1,
                                             collect_critiques=critiques)
    instance.client.chat.completions.create.side_effect = list(map(completion, contents))
    return instance


EXAMPLE = {"attribute": "concise", "query": "What happened?", "enhanced_query": "What happened? Be concise.",
           "response_1": "The test passed. Extra detail follows.", "response_2": "It passed after a brief delay."}


class IncrementTests(unittest.TestCase):
    def test_shared_generation_and_rm_evaluation_sources(self):
        project = Path(__file__).resolve().parents[1]
        upstream = project.parent / "T-POP-main"
        if not upstream.is_dir():
            self.skipTest("The upstream sibling is only needed for this byte comparison")
        for name in ("model.py", "rm_eval.py"):
            self.assertEqual((project / name).read_bytes(), (upstream / name).read_bytes())

    def test_critique_keeps_choice_call_and_edits_selected_span_only(self):
        control = collector(["[[B]]"], critiques=False)
        intervention = collector(["[[B]]", json.dumps({"comment": "Drop the time detail.",
                                                   "original_span": " after a brief delay", "replacement": ""})],
                                 critiques=True)
        normal = control.collect_single(EXAMPLE)
        extended = intervention.collect_single(EXAMPLE)
        self.assertEqual({key: extended[key] for key in normal}, normal)
        self.assertEqual(normal["label"], 1)
        self.assertEqual(control.client.chat.completions.create.call_args,
                         intervention.client.chat.completions.create.call_args_list[0])
        self.assertEqual(control.client.chat.completions.create.call_count, 1)
        self.assertEqual(extended["critique"]["revised_y"], "What happened? Be concise. It passed.")

    def test_unusable_edit_and_optional_api_failure_keep_binary_label(self):
        self.assertIsNone(judger.local_revision("short short answer", "short", "long"))
        self.assertIsNone(judger.local_revision("short answer", "short answer", "different answer"))
        instance = collector(["[[A]]", '{"comment":"Explain the result.","original_span":"absent","replacement":"x"}'],
                             critiques=True)
        result = instance.collect_single(EXAMPLE)
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["critique"]["status"], "comment_only")
        self.assertNotIn("revised_y", result["critique"])
        failed = collector(["[[A]]", "invalid JSON"], critiques=True).collect_single(EXAMPLE)
        self.assertEqual(failed["label"], 0)
        self.assertEqual(failed["critique"]["status"], "failed")

    def test_bce_uses_upstream_label_direction(self):
        a = torch.zeros(2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)
        imple_armo.pairwise_loss(a, b, torch.tensor([0, 1])).backward()
        self.assertLess(a.grad[0].item(), 0)
        self.assertLess(b.grad[1].item(), 0)

    def test_loss_compares_edit_to_winner_and_uses_original_batch_size(self):
        generator = SimpleNamespace(get_embedding=lambda texts: torch.tensor([[3.0]]), reward_model=torch.nn.Identity())
        a, b = torch.tensor([9.0, 4.0]), torch.tensor([1.0, 2.0])
        labels = torch.tensor([1, 0])
        loss = imple_armo.critique_loss(generator, a, b, labels, ["edit", ""])
        self.assertAlmostEqual(loss.item(), torch.nn.functional.softplus(torch.tensor(-2.0)).item() / 2)
        self.assertIsNone(imple_armo.critique_loss(generator, a, b, labels, ["", ""]))

    def test_training_uses_same_dual_arm_call_and_step_schedule(self):
        class Generator:
            def __init__(self):
                self.reward_model = torch.nn.Linear(1, 1)
                self.tokenizer = SimpleNamespace(decode=lambda reply, **kwargs: reply)
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                prefix = kwargs["prompt"]
                return prefix + " The test passed. Extra detail follows.", prefix + " It failed."

            def get_embedding(self, texts):
                return torch.tensor([[len(text.split()) / 10] for text in texts])

        class CountedAdamW(torch.optim.AdamW):
            steps = 0

            def step(self, *args, **kwargs):
                CountedAdamW.steps += 1
                return super().step(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch.object(imple_armo.torch.optim, "AdamW", CountedAdamW):
            results = {}
            for mode in ("choice", "critique"):
                torch.manual_seed(42)
                generator = Generator()
                responses = ["[[A]]"] if mode == "choice" else [
                    "[[A]]", '{"comment":"Remove redundant detail.","original_span":" Extra detail follows.","replacement":""}']
                judge = collector(responses * 3, critiques=mode == "critique")
                CountedAdamW.steps = 0
                cfg = SimpleNamespace(output_dir=str(Path(folder) / mode), feedback_mode=mode, critique_weight=4.0,
                                      batch_size=2, lr=0.01, epochs_per_query=1, reward_model_device="cpu",
                                      attribute="concise", reward_weight=1.0, max_new_tokens=12,
                                      pre_screen_beam_width=5, nu_mab=1.0, lambda_mab=1.0)
                imple_armo.train_reward(generator, judge, "concise", ["Q1", "Q2", "Q3"], cfg)
                results[mode] = (generator.calls, CountedAdamW.steps, generator.reward_model.weight.detach().clone())
            self.assertEqual(results["choice"][:2], results["critique"][:2])
            self.assertEqual(results["choice"][1], 2)
            self.assertFalse(torch.equal(results["choice"][2], results["critique"][2]))

    def test_optional_evaluation_feedback_does_not_change_winner(self):
        evaluation = WinrateEvaluator(
            collector(["[[A]]", '{"comment":"Good enough for me.","original_span":"","replacement":""}'],
                      critiques=True), "concise")
        ours, baseline = ({"query": "q", "response": "A reply"}, {"query": "q", "response": "B reply"})
        stats = evaluation.evaluate_all([(ours, baseline)])
        self.assertEqual(stats["wins"], 1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "wins.txt"
            evaluation.save_results(str(output), stats, "a.jsonl", "b.jsonl")
            feedback = json.loads(output.with_suffix(".critiques.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(feedback["critique"]["comment"], "Good enough for me.")


if __name__ == "__main__":
    unittest.main()
