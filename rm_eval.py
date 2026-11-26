import argparse
import json
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preference_attributes import PREFERENCE_ATTRIBUTES


DEFAULT_MODEL_PATH = "/data/zhujiajun/schema-mem/ArmoRM-Llama3-8B-v0.1"
SCORE_FIELD = "armorm_instruction_following"


@dataclass(frozen=True)
class Question:
    question_id: int | str
    query: str


def _read_rows(path):
    """Read JSONL or arrays, including the concatenated arrays in older outputs."""
    text = Path(path).read_text(encoding="utf-8-sig")
    remaining = text.lstrip()
    if not remaining.startswith("["):
        for line_number, line in enumerate(text.split("\n"), 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error.msg}") from error
        return

    decoder = json.JSONDecoder()
    position = len(text) - len(remaining)
    newlines = [position for position, char in enumerate(text) if char == "\n"]

    def skip_whitespace(position):
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    while position < len(text):
        line_number = bisect_left(newlines, position) + 1
        if text[position] != "[":
            raise ValueError(f"{path}:{line_number}: expected a JSON array")
        position = skip_whitespace(position + 1)
        if text[position:position + 1] != "]":
            while True:
                line_number = bisect_left(newlines, position) + 1
                try:
                    row, position = decoder.raw_decode(text, position)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{error.lineno}: {error.msg}") from error
                yield line_number, row
                position = skip_whitespace(position)
                delimiter = text[position:position + 1]
                if delimiter == "]":
                    break
                if delimiter != ",":
                    line_number = bisect_left(newlines, position) + 1
                    raise ValueError(f"{path}:{line_number}: expected ',' or ']'")
                position = skip_whitespace(position + 1)
        position = skip_whitespace(position + 1)


def _question_id(value, location):
    if type(value) is int or (isinstance(value, str) and value.strip()):
        return value
    raise ValueError(f"{location}: question ID must be an integer or a nonempty string")


def load_questions(path):
    questions = []
    question_ids = set()
    for position, (line_number, row) in enumerate(_read_rows(path)):
        location = f"{path}:{line_number}"
        query = row.get("question") if isinstance(row, dict) else row
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{location}: expected a nonempty question")
        index = row.get("index", position) if isinstance(row, dict) else position
        question_id = _question_id(index, location)
        if question_id in question_ids:
            raise ValueError(f"{location}: duplicate question ID {question_id!r}")
        question_ids.add(question_id)
        questions.append(Question(question_id, query))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions


def load_responses(path, questions):
    by_id = {question.question_id: question for question in questions}
    by_query = defaultdict(list)
    for question in questions:
        by_query[question.query].append(question.question_id)

    answers = {}
    for line_number, row in _read_rows(path):
        location = f"{path}:{line_number}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a response object")
        if "question_id" in row:
            question_id = _question_id(row["question_id"], location)
            if question_id not in by_id:
                raise ValueError(f"{location}: unknown question ID {question_id!r}")
            if "query" in row and row["query"] != by_id[question_id].query:
                raise ValueError(f"{location}: query conflicts with question ID {question_id!r}")
        else:
            query = row.get("query")
            matches = by_query.get(query, []) if isinstance(query, str) else []
            if len(matches) != 1:
                reason = "matches multiple questions" if matches else "does not match a question"
                raise ValueError(f"{location}: query {reason}")
            question_id = matches[0]
        if question_id in answers:
            raise ValueError(f"{location}: duplicate response for question ID {question_id!r}")
        reply = row.get("ours", row.get("base"))
        if reply is not None and not isinstance(reply, str):
            raise ValueError(f"{location}: response must be a string or null")
        answers[question_id] = reply
    return answers


def make_score_rows(questions, answers):
    rows = []
    for question in questions:
        response = answers.get(question.question_id)
        if response is not None and not response.strip():
            response = None
        rows.append({
            "question_id": question.question_id,
            "query": question.query,
            "response": response,
            "status": "missing_response" if response is None else "score_failed",
            SCORE_FIELD: None,
        })
    return rows


def summarize_scores(rows):
    scores = [row[SCORE_FIELD] for row in rows if row["status"] == "scored"]
    missing = sum(row["status"] == "missing_response" for row in rows)
    planned = len(rows)
    with_response = planned - missing
    return {
        "planned": planned,
        "with_response": with_response,
        "response_coverage": with_response / planned if planned else None,
        "missing_responses": missing,
        "scored": len(scores),
        "score_coverage": len(scores) / planned if planned else None,
        "score_failures": with_response - len(scores),
        "mean": fmean(scores) if scores else None,
    }


def _bootstrap_interval(deltas):
    if len(deltas) < 2:
        return None
    rng = np.random.default_rng(42)
    means = np.empty(10_000)
    for start in range(0, len(means), 100):
        means[start:start + 100] = rng.choice(deltas, size=(100, len(deltas))).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare_scores(baseline, candidate):
    rows = []
    deltas = []
    common_responses = 0
    for left, right in zip(baseline, candidate, strict=True):
        if (left["question_id"], left["query"]) != (right["question_id"], right["query"]):
            raise ValueError("Comparison rows must be aligned by question ID and query")
        baseline_score = left[SCORE_FIELD]
        candidate_score = right[SCORE_FIELD]
        delta = None
        if baseline_score is not None and candidate_score is not None:
            delta = candidate_score - baseline_score
            deltas.append(delta)
        common_responses += left["response"] is not None and right["response"] is not None
        rows.append({
            "question_id": left["question_id"],
            "query": left["query"],
            "baseline": {field: left[field] for field in ("response", "status", SCORE_FIELD)},
            "candidate": {field: right[field] for field in ("response", "status", SCORE_FIELD)},
            "delta": delta,
        })
    return rows, {
        "common_responses": common_responses,
        "common_scored": len(deltas),
        "paired_mean_delta": fmean(deltas) if deltas else None,
        "paired_delta_ci_95": _bootstrap_interval(deltas),
    }


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


class PersonalizationRMEvaluator:
    def __init__(self, attribute: str, device: str, rm_batch_size: int = 128, model_path=DEFAULT_MODEL_PATH):
        if rm_batch_size < 1:
            raise ValueError("ArmoRM batch size must be positive")
        self.attribute = attribute
        self.device = device
        self.rm_batch_size = rm_batch_size
        self.rmodel = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    def load_data(self, file_path):
        return [row for _, row in _read_rows(file_path)]

    def batch_score(self, messages):
        batch_input = [
            self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        batch_inputs = self.tokenizer(
            batch_input, padding=True, truncation=True, add_special_tokens=True,
            max_length=1024, return_tensors="pt",
        )

        input_ids = batch_inputs.input_ids.to(self.device)
        attention_mask = batch_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            output = self.rmodel(input_ids=input_ids, attention_mask=attention_mask)
            return output.rewards[:, 6].cpu().float().tolist()

    def divide_data_batch(self, dataset):
        return [dataset[start:start + self.rm_batch_size] for start in range(0, len(dataset), self.rm_batch_size)]

    def score_groups(self, *groups):
        preference = PREFERENCE_ATTRIBUTES.get(
            self.attribute, f"Your answer should be {self.attribute} as much as possible."
        )
        inputs = list(dict.fromkeys(
            (row["query"], row["response"])
            for group in groups for row in group if row["response"] is not None
        ))
        scores = {}
        for batch in tqdm(self.divide_data_batch(inputs), desc="Scoring responses"):
            messages = [
                [
                    {"role": "user", "content": f"{query} {preference}"},
                    {"role": "assistant", "content": response},
                ]
                for query, response in batch
            ]
            batch_scores = self.batch_score(messages)
            if len(batch_scores) != len(batch):
                raise RuntimeError(f"ArmoRM returned {len(batch_scores)} scores for {len(batch)} responses")
            for key, score in zip(batch, batch_scores):
                scores[key] = float(score) if math.isfinite(score) else None

        scored_groups = []
        for group in groups:
            scored_rows = []
            for row in group:
                score = scores.get((row["query"], row["response"]))
                scored_rows.append({
                    **row,
                    "status": "scored" if score is not None else row["status"],
                    SCORE_FIELD: score,
                })
            scored_groups.append(scored_rows)
        return scored_groups

    def get_rm_eval(self, response_file: str, response_key: str = "ours"):
        questions = []
        responses = {}
        for index, row in enumerate(self.load_data(response_file)):
            if isinstance(row, dict) and isinstance(row.get("query"), str) and response_key in row:
                questions.append(Question(index, row["query"]))
                responses[index] = row[response_key]
        (scored,) = self.score_groups(make_score_rows(questions, responses))
        mean = summarize_scores(scored)["mean"]
        return mean if mean is not None else math.nan


def score_saved_responses(args):
    if args.batch_size < 1:
        raise ValueError("ArmoRM batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    questions = load_questions(args.data_file)
    candidate = make_score_rows(questions, load_responses(args.responses_file, questions))
    groups = [candidate]
    if args.baseline_file is not None:
        baseline = make_score_rows(questions, load_responses(args.baseline_file, questions))
        groups.insert(0, baseline)

    if any(row["response"] is not None for group in groups for row in group):
        evaluator = PersonalizationRMEvaluator(args.attribute, args.device, args.batch_size, args.model_path)
        groups = evaluator.score_groups(*groups)
        del evaluator

    summary = {
        "attribute": args.attribute,
        "metric": SCORE_FIELD,
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "device": args.device,
        "batch_size": args.batch_size,
        "data_file": str(args.data_file),
        "responses_file": str(args.responses_file),
        "candidate": summarize_scores(groups[-1]),
    }
    if len(groups) == 2:
        baseline, candidate = groups
        comparisons, paired = compare_scores(baseline, candidate)
        summary.update(baseline_file=str(args.baseline_file), baseline=summarize_scores(baseline), **paired)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if len(groups) == 2:
        for name, rows in zip(("baseline", "candidate"), groups):
            directory = args.output_dir / name
            directory.mkdir()
            write_jsonl(directory / "scores.jsonl", rows)
        write_jsonl(args.output_dir / "comparisons.jsonl", comparisons)
    else:
        write_jsonl(args.output_dir / "scores.jsonl", groups[0])
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score existing responses with ArmoRM")
    subcommands = parser.add_subparsers(dest="command", required=True)
    score = subcommands.add_parser("score", help="Score one run or compare two runs")
    score.add_argument("--data-file", type=Path, required=True)
    score.add_argument("--responses-file", type=Path, required=True)
    score.add_argument("--baseline-file", type=Path)
    score.add_argument("--attribute", required=True)
    score.add_argument("--model-path", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        score_saved_responses(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"ArmoRM scoring failed: {error}\n")
    print(f"Saved ArmoRM results to {args.output_dir}")


if __name__ == "__main__":
    main()
