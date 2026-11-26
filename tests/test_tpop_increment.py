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

    def test_critique_keeps_choice_call_and_accepts_only_user_comment(self):
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
        self.assertEqual(extended["critique"], {"comment": "Drop the time detail.", "status": "comment_only"})

    def test_empty_comment_and_optional_api_failure_keep_binary_label(self):
        instance = collector(["[[A]]", '{"comment":""}'], critiques=True)
        result = instance.collect_single(EXAMPLE)
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["critique"]["status"], "empty")
        self.assertNotIn("revised_y", result["critique"])
        failed = collector(["[[A]]", "invalid JSON"], critiques=True).collect_single(EXAMPLE)
        self.assertEqual(failed["label"], 0)
        self.assertEqual(failed["critique"]["status"], "failed")

    def test_revision_is_generated_by_frozen_base_model_from_question_answer_and_comment(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            chat_template = "chat"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_options = kwargs
                return "templated prompt"

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                self.tokenizer_options = kwargs
                return Batch(input_ids=torch.tensor([[2, 4, 6]]), attention_mask=torch.ones(1, 3))

            def decode(self, ids, **kwargs):
                self.decoded_ids = ids.tolist()
                return " It passed. "

        def generate(**kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["max_new_tokens"], 2)
            return torch.tensor([[2, 4, 6, 7, 8]])

        tokenizer = Tokenizer()
        generator = SimpleNamespace(tokenizer=tokenizer, llm_device="cpu",
                                    LLM=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5),
                                                        generate=generate))
        revised = imple_armo.generate_revision(generator, "What happened?", "It passed after a brief delay.",
                                               "Drop the time detail.", 8)
        self.assertEqual(revised, "It passed.")
        self.assertEqual(tokenizer.decoded_ids, [7, 8])
        self.assertTrue(tokenizer.template_options["add_generation_prompt"])
        self.assertFalse(tokenizer.tokenizer_options["add_special_tokens"])
        item = json.loads(tokenizer.messages[0]["content"].split("\n\n", 1)[1])
        self.assertEqual(item, {"question": "What happened?", "selected_answer": "It passed after a brief delay.",
                                "user_feedback": "Drop the time detail."})

    def test_unusable_assistant_revision_preserves_comment_and_excludes_legacy_edits(self):
        for result, status in (("original", "revision_unusable"), ("", "revision_unusable"),
                               (RuntimeError("backend error"), "revision_failed")):
            critique = {"comment": "Explain the result.", "status": "comment_only"}
            with patch.object(imple_armo, "generate_revision", side_effect=[result]):
                imple_armo.add_assistant_revision(None, critique, "question", "original", 8)
            self.assertEqual(critique, {"comment": "Explain the result.", "status": status})

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preferences.jsonl"
            original = {"y1": "A", "y2": "B", "label": 0}
            legacy = {**original, "critique": {"status": "usable", "revised_y": "user-generated rewrite"}}
            current = {**original, "critique": {"status": "usable", "revision_source": "base_llm",
                                               "revised_y": "assistant rewrite"}}
            path.write_text("\n".join(map(json.dumps, (legacy, current))), encoding="utf-8")
            rows = imple_armo.PreferenceDataset(str(path), include_critiques=True)
            self.assertEqual(rows[0], ("A", "B", 0, ""))
            self.assertEqual(rows[1], ("A", "B", 0, "assistant rewrite"))

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
                responses = ["[[A]]"] if mode == "choice" else ["[[A]]", '{"comment":"Remove redundant detail."}']
                judge = collector(responses * 3, critiques=mode == "critique")
                CountedAdamW.steps = 0
                cfg = SimpleNamespace(output_dir=str(Path(folder) / mode), feedback_mode=mode, critique_weight=4.0,
                                      batch_size=2, lr=0.01, epochs_per_query=1, reward_model_device="cpu",
                                      attribute="concise", reward_weight=1.0, max_new_tokens=12,
                                      pre_screen_beam_width=5, nu_mab=1.0, lambda_mab=1.0)
                with patch.object(imple_armo, "generate_revision", return_value="The test passed.") as revise:
                    imple_armo.train_reward(generator, judge, "concise", ["Q1", "Q2", "Q3"], cfg)
                self.assertEqual(revise.call_count, 3 if mode == "critique" else 0)
                if mode == "critique":
                    self.assertEqual(revise.call_args.args[2:], ("The test passed. Extra detail follows.",
                                                                "Remove redundant detail.", 12))
                    log = Path(cfg.output_dir) / "preferences" / "preferences.jsonl"
                    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
                    self.assertEqual(record["critique"]["revision_source"], "base_llm")
                    self.assertTrue(record["critique"]["revised_y"].endswith("The test passed."))
                results[mode] = (generator.calls, CountedAdamW.steps, generator.reward_model.weight.detach().clone())
            self.assertEqual(results["choice"][:2], results["critique"][:2])
            self.assertEqual(results["choice"][1], 2)
            self.assertFalse(torch.equal(results["choice"][2], results["critique"][2]))

    def test_optional_evaluation_feedback_does_not_change_winner(self):
        evaluation = WinrateEvaluator(
            collector(["[[A]]", '{"comment":"Good enough for me."}'],
                      critiques=True), "concise")
        ours, baseline = ({"query": "q", "response": "A reply"}, {"query": "q", "response": "B reply"})
        stats = evaluation.evaluate_all([(ours, baseline)])
        self.assertEqual(stats["wins"], 1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "wins.txt"
            evaluation.save_results(str(output), stats, "a.jsonl", "b.jsonl")
            feedback = json.loads(output.with_suffix(".critiques.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(feedback["critique"]["comment"], "Good enough for me.")
            self.assertNotIn("revised_answer", feedback["critique"])


    def test_low_memory_uses_cpu_and_failure_is_not_written_as_zero(self):
        self.assertEqual(imple_armo.scoring_device("cuda:4", 1024), "cpu")
        self.assertEqual(imple_armo.scoring_device("cuda:4", 20 * 1024 ** 3), "cuda:4")
        self.assertTrue(imple_armo.is_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))

        calls = []

        def fake_score(responses_file, attribute, device, batch_size):
            calls.append((device, batch_size))
            if device.startswith("cuda"):
                raise RuntimeError("CUDA out of memory")
            return 0.25

        with patch.object(imple_armo, "_free_cuda_bytes", return_value=1024), \
                patch.object(imple_armo, "_score_with_armorm", side_effect=fake_score):
            score, device, batch_size, note = imple_armo.score_responses("responses.jsonl", "creative", "cuda:4", 128)
            self.assertEqual((score, device, batch_size), (0.25, "cpu", 4))
            self.assertIn("below the 15 GiB", note)
            self.assertEqual(calls, [("cpu", 4)])

        calls.clear()
        with patch.object(imple_armo, "_free_cuda_bytes", return_value=20 * 1024 ** 3), \
                patch.object(imple_armo, "_score_with_armorm", side_effect=fake_score):
            score, device, batch_size, note = imple_armo.score_responses("responses.jsonl", "creative", "cuda:4", 128)
            self.assertEqual((score, device, batch_size), (0.25, "cpu", 4))
            self.assertIn("GPU scoring failed", note)
            self.assertEqual(calls, [("cuda:4", 128), ("cpu", 4)])

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.txt"
            imple_armo.append_experiment_results(str(path), {
                "error": "CUDA out of memory\nRuntimeError: allocation failed",
                "total": 3,
                "failed_generations": 1,
            }, 100, 548)
            text = path.read_text(encoding="utf-8")
            self.assertIn("ArmoRM scoring failed", text)
            self.assertIn("allocation failed", text)
            self.assertNotIn("ArmoRM Score:", text)
            self.assertNotIn("0.000000", text)


if __name__ == "__main__":
    unittest.main()
