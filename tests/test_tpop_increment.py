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
import rm_eval
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
    def test_full_experiment_writes_scores_and_keeps_result_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            data_file = Path(folder) / "questions.json"
            data_file.write_text(json.dumps([
                {"index": 7, "question": "First question"},
                {"index": "q9", "question": "Second question"},
                {"index": 21, "question": "Third question"},
            ]), encoding="utf-8")
            output = Path(folder) / "run"
            cfg = SimpleNamespace(
                output_dir=str(output), data_file=str(data_file), llm_path="mock-llm",
                seed=42, attribute="creative", feedback_mode="choice", critique_weight=0.5,
                openai_api_key="unused", preference_model_name="unused", preference_api_base_url="unused",
                preference_api_sleep_time=0, preference_api_max_retries=1, embedding_model_path="mock-embedding",
                llm_device="cpu", reward_model_device="cpu", embedding_model_device="cpu",
                normalize_sentence_embeddings=True, train_samples=1, lr=0.001,
                batch_size=2, rm_batch_size=128, epochs_per_query=1,
                reward_weight=1.0, max_new_tokens=128, pre_screen_beam_width=40, nu_mab=0.5, lambda_mab=1.0,
            )
            evaluator = fake_evaluator({"First answer": 0.7, "Third answer": 0.3}, batch_size=128)
            with patch.multiple(
                imple_armo, PreferenceCollector=Mock(), MABRewardGenerator=Mock(return_value=SimpleNamespace()),
                train_reward=Mock(return_value=str(output / "unused.pt")),
                generate_with_fixed_weight=Mock(side_effect=["First answer", "", "Third answer"]),
                PersonalizationRMEvaluator=Mock(return_value=evaluator),
            ):
                imple_armo.run(cfg)
            records = read_jsonl(output / "responses" / "responses_armo_rw_1.00.jsonl")
            self.assertEqual(records, [
                {"question_id": 7, "query": "First question", "ours": "First answer"},
                {"question_id": 21, "query": "Third question", "ours": "Third answer"},
            ])
            scores = read_jsonl(output / "scores.jsonl")
            self.assertEqual([row["question_id"] for row in scores], [7, "q9", 21])
            self.assertEqual([row["status"] for row in scores], ["scored", "missing_response", "scored"])
            self.assertEqual([row[rm_eval.SCORE_FIELD] for row in scores], [0.7, None, 0.3])
            result = (output / "result.txt").read_text(encoding="utf-8")
            for line in (
                "ArmoRM Score: 0.500000", "ArmoRM Device: cpu", "ArmoRM Batch Size: 128",
                "Total Responses: 2", "Evaluation Queries: 3", "Failed Generations: 1",
                "ArmoRM Scored Responses: 2", "ArmoRM Score Failures: 0",
            ):
                self.assertIn(line, result)

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
                loaded = []

                class RecordingDataset(imple_armo.PreferenceDataset):
                    def __init__(self, path, include_critiques=False):
                        super().__init__(path, include_critiques=include_critiques)
                        loaded.append((len(self), len(generator.calls)))

                with patch.object(imple_armo, "PreferenceDataset", RecordingDataset), \
                        patch.object(imple_armo, "generate_revision", return_value="The test passed.") as revise:
                    imple_armo.train_reward(generator, judge, "concise", ["Q1", "Q2", "Q3"], cfg)
                # One sample, then two, then all three after the last generation.
                self.assertEqual(loaded, [(1, 1), (2, 2), (3, 3)])
                saved = torch.load(Path(cfg.output_dir) / "checkpoints" / "final_model.pt",
                                   map_location="cpu", weights_only=True)
                self.assertTrue(torch.equal(saved["weight"], generator.reward_model.weight.detach()))
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
            # Fits on 1, 2, and finally 3 samples. The last fit has two batches.
            self.assertEqual(results["choice"][1], 4)
            self.assertFalse(torch.equal(results["choice"][2], results["critique"][2]))

            torch.manual_seed(42)
            generator = Generator()
            judge = collector(["[[A]]", "[[A]]"], critiques=False)
            real_collect = judge.collect_single
            attempts = {"n": 0}

            def collect(example):
                attempts["n"] += 1
                if attempts["n"] == 3:
                    return None
                return real_collect(example)

            judge.collect_single = collect
            CountedAdamW.steps = 0
            cfg = SimpleNamespace(output_dir=str(Path(folder) / "missing-last"), feedback_mode="choice",
                                  critique_weight=4.0, batch_size=2, lr=0.01, epochs_per_query=1,
                                  reward_model_device="cpu", attribute="concise", reward_weight=1.0,
                                  max_new_tokens=12, pre_screen_beam_width=5, nu_mab=1.0, lambda_mab=1.0)
            imple_armo.train_reward(generator, judge, "concise", ["Q1", "Q2", "Q3"], cfg)
            preferences = Path(cfg.output_dir) / "preferences" / "preferences.jsonl"
            self.assertEqual(len(preferences.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(generator.calls), 3)
            self.assertEqual(CountedAdamW.steps, 2)

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

        def fake_score(rows, attribute, device, batch_size):
            calls.append((device, batch_size))
            if device.startswith("cuda"):
                raise RuntimeError("CUDA out of memory")
            return [{"armorm_instruction_following": 0.25}]

        with patch.object(imple_armo, "_free_cuda_bytes", return_value=1024), \
                patch.object(imple_armo, "_score_with_armorm", side_effect=fake_score):
            rows, device, batch_size, note = imple_armo.score_responses([], "creative", "cuda:4", 128)
            self.assertEqual((rows, device, batch_size), ([{"armorm_instruction_following": 0.25}], "cpu", 4))
            self.assertIn("below the 15 GiB", note)
            self.assertEqual(calls, [("cpu", 4)])

        calls.clear()
        with patch.object(imple_armo, "_free_cuda_bytes", return_value=20 * 1024 ** 3), \
                patch.object(imple_armo, "_score_with_armorm", side_effect=fake_score):
            rows, device, batch_size, note = imple_armo.score_responses([], "creative", "cuda:4", 128)
            self.assertEqual((rows, device, batch_size), ([{"armorm_instruction_following": 0.25}], "cpu", 4))
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

    def test_full_evaluation_preserves_rows_when_scorer_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = SimpleNamespace(output_dir=folder, reward_weight=1.0, nu_mab=0.5, lambda_mab=1.0, llm_device="cpu")
            questions = [rm_eval.Question(7, "First question"), rm_eval.Question(9, "Second question")]
            with patch.multiple(
                imple_armo, generate_with_fixed_weight=Mock(side_effect=["First answer", ""]),
                score_responses=Mock(side_effect=RuntimeError("scoring unavailable")),
            ):
                result = imple_armo.evaluate(SimpleNamespace(), questions, "creative", 128, 40, cfg)
            rows = read_jsonl(Path(folder) / "scores.jsonl")
            self.assertEqual([row["status"] for row in rows], ["score_failed", "missing_response"])
            self.assertEqual([row[rm_eval.SCORE_FIELD] for row in rows], [None, None])
            self.assertEqual(result["score_failures"], 1)
            self.assertEqual(result["failed_generations"], 1)
            self.assertIsNone(result["avg_reward"])
            self.assertIn("scoring unavailable", result["error"])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_evaluator(values, batch_size=4):
    evaluator = object.__new__(rm_eval.PersonalizationRMEvaluator)
    evaluator.attribute = "creative"
    evaluator.rm_batch_size = batch_size
    evaluator.batch_score = Mock(side_effect=lambda messages: [values[pair[1]["content"]] for pair in messages])
    return evaluator


class ArmoRMTests(unittest.TestCase):
    def test_legacy_mean_and_bounded_batches(self):
        evaluator = fake_evaluator({"one": 0.2, "two": 0.5, "three": 0.8}, batch_size=2)
        self.assertEqual(evaluator.divide_data_batch([1]), [[1]])
        self.assertEqual(evaluator.divide_data_batch([1, 2, 3]), [[1, 2], [3]])
        self.assertEqual(evaluator.divide_data_batch([]), [])
        with tempfile.TemporaryDirectory() as folder:
            responses = Path(folder) / "legacy.json"
            responses.write_text(
                '[{"query":"q1","base":"one"}, {"query":"q2","base":"two"}]\n'
                '[{"query":"q3","base":"three"}]\n', encoding="utf-8",
            )
            self.assertAlmostEqual(evaluator.get_rm_eval(responses, response_key="base"), 0.5)
        calls = evaluator.batch_score.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [2, 1])
        self.assertEqual(calls[0].args[0][0][0], {
            "role": "user", "content": f"q1 {judger.PREFERENCE_ATTRIBUTES['creative']}",
        })

    def test_response_errors_report_file_and_line(self):
        questions = [rm_eval.Question(1, "duplicate text"), rm_eval.Question(2, "duplicate text")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "responses.jsonl"
            good = {"question_id": 1, "query": "duplicate text", "ours": "a"}
            for row, message in [
                ({"query": "duplicate text", "base": "b"}, "matches multiple questions"),
                ({"query": "missing", "ours": "b"}, "does not match a question"),
                ({"question_id": 99, "ours": "b"}, "unknown question ID"),
                ({"question_id": 2, "query": "wrong", "ours": "b"}, "query conflicts"),
                (good, "duplicate response"),
            ]:
                with self.subTest(message=message):
                    path.write_text(json.dumps(good) + "\n\n" + json.dumps(row), encoding="utf-8")
                    with self.assertRaises(ValueError) as error:
                        rm_eval.load_responses(path, questions)
                    self.assertIn(f"{path}:3:", str(error.exception))
                    self.assertIn(message, str(error.exception))
            path.write_text('[\n  ' + json.dumps(good) + ',\n  ' + json.dumps(good) + '\n]', encoding="utf-8")
            with self.assertRaises(ValueError) as error:
                rm_eval.load_responses(path, questions)
            self.assertIn(f"{path}:3: duplicate response", str(error.exception))

    def test_cli_pairs_by_id_reuses_scores_and_reports_missing_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            questions = root / "data.json"
            questions.write_text('["q1", "q2", "q3", "q4"]', encoding="utf-8")
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            rm_eval.write_jsonl(baseline_path, [
                {"query": "q3", "base": "baseline only"},
                {"query": "q1", "base": "same"},
                {"query": "q4", "base": "valid"},
                {"query": "q2", "base": "low"},
            ])
            rm_eval.write_jsonl(candidate_path, [
                {"question_id": 3, "query": "q4", "ours": "nonfinite"},
                {"question_id": 1, "query": "q2", "ours": "high"},
                {"question_id": 0, "query": "q1", "ours": "same"},
            ])
            evaluator = fake_evaluator({
                "same": 0.3, "low": 0.1, "high": 0.5, "baseline only": 0.4, "valid": 0.2, "nonfinite": float("nan"),
            })
            output = root / "output"
            command = [
                "score", "--data-file", str(questions), "--responses-file", str(candidate_path),
                "--baseline-file", str(baseline_path), "--attribute", "creative",
                "--model-path", "mock-model", "--output-dir", str(output),
            ]
            with patch.object(rm_eval, "PersonalizationRMEvaluator", return_value=evaluator) as load:
                rm_eval.main(command)
                load.assert_called_once_with("creative", "cpu", 4, "mock-model")
            self.assertEqual([len(call.args[0]) for call in evaluator.batch_score.call_args_list], [4, 2])
            comparison = read_jsonl(output / "comparisons.jsonl")
            self.assertEqual([row["question_id"] for row in comparison], [0, 1, 2, 3])
            self.assertEqual([row["delta"] for row in comparison], [0, 0.4, None, None])
            self.assertEqual(comparison[2]["candidate"]["status"], "missing_response")
            self.assertEqual(comparison[3]["candidate"]["status"], "score_failed")
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["baseline"]["scored"], 4)
            self.assertEqual(report["candidate"]["missing_responses"], 1)
            self.assertEqual(report["candidate"]["score_failures"], 1)
            self.assertEqual(report["common_responses"], 3)
            self.assertEqual(report["common_scored"], 2)
            self.assertAlmostEqual(report["paired_mean_delta"], 0.2)
            self.assertEqual(report["paired_delta_ci_95"], [0.0, 0.4])
            for name in ("baseline", "candidate"):
                scores = read_jsonl(output / name / "scores.jsonl")
                self.assertEqual(report[name], rm_eval.summarize_scores(scores))

    def test_single_run_output_and_insufficient_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_file = root / "questions.json"
            data_file.write_text('[{"index":"A","question":"first"},{"index":4,"question":"second"}]', encoding="utf-8")
            responses = root / "responses.jsonl"
            rm_eval.write_jsonl(responses, [{"query": "first", "ours": "answer"}])
            output = root / "result"
            with patch.object(rm_eval, "PersonalizationRMEvaluator", return_value=fake_evaluator({"answer": 0.5})):
                rm_eval.main([
                    "score", "--data-file", str(data_file), "--responses-file", str(responses),
                    "--attribute", "creative", "--model-path", "mock-model", "--output-dir", str(output),
                ])
            scores = read_jsonl(output / "scores.jsonl")
            self.assertEqual([row["question_id"] for row in scores], ["A", 4])
            self.assertEqual(scores[1]["status"], "missing_response")
            _, summary = rm_eval.compare_scores(scores, scores)
            self.assertEqual(summary["common_scored"], 1)
            self.assertIsNone(summary["paired_delta_ci_95"])


if __name__ == "__main__":
    unittest.main()
