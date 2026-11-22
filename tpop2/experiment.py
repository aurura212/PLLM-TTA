"""Run the matched experiments and rebuild their summaries from saved records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np

from .data import Task, profiles, stable_seed
from .feedback import InvalidFeedback
from .reward import ATTRIBUTES, eligible_constraints, extract_features, fit_reward, predict_choice

GROUPS = ("choice", "reason", "critique")
SEEDS = (17, 29, 43)
CHECKPOINTS = (0, 4, 8, 16)
COMMENT_ROUNDS = (4, 8, 12, 16)
PAIR_TYPES = ("missing_only", "observed_only", "tradeoff", "both_change")


def write_text(path: Path, text: str) -> None:
    """Publish completed artifacts atomically, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_fingerprint(tasks: list[Task]) -> str:
    text = json.dumps([vars(task) for task in tasks], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def target_pair(profile: dict) -> tuple[int, int]:
    return int(profile["lead"]), int(profile["example"])


def _flip(features: tuple[int, int], index: int) -> tuple[int, int]:
    return tuple(1 - value if i == index else value for i, value in enumerate(features))


def orient(a, b, seed: int):
    """Keep variant presentation unrelated to experiment group or winner."""
    return (a, b) if random.Random(seed).getrandbits(1) else (b, a)


def training_pair(profile: dict, seed: int) -> tuple[tuple[int, int], tuple[int, int], str]:
    """One axis is held fixed at the same flawed value throughout one user's tape."""
    observed = ATTRIBUTES.index(profile["priority"])
    missing = 1 - observed
    shared_flaw = _flip(target_pair(profile), missing)
    a, b = orient(shared_flaw, _flip(shared_flaw, observed), seed)
    return a, b, ATTRIBUTES[missing]


def test_pair(profile: dict, kind: str, seed: int) -> tuple[tuple[int, int], tuple[int, int]]:
    desired = target_pair(profile)
    observed = ATTRIBUTES.index(profile["priority"])
    missing = 1 - observed
    lacking_observed = _flip(desired, observed)
    lacking_missing = _flip(desired, missing)
    variants = {
        "missing_only": (desired, lacking_missing),
        "observed_only": (desired, lacking_observed),
        "tradeoff": (lacking_missing, lacking_observed),
        "both_change": (desired, _flip(lacking_missing, observed)),
    }
    return orient(*variants[kind], seed)


def _score(prediction: str, target: str) -> float:
    return 0.5 if prediction == "tie" else float(prediction == target)


def _asks_record(kind, client, profile, task, round_row, seed, context) -> dict:
    a, b, choice = round_row["a"], round_row["b"], round_row["choice"]
    comment = client.comment(kind, profile, task.question, a, b, choice, seed, context)
    claims, parse_error = [], None
    if comment.strip():
        try:
            claims = client.parse_comment(task.question, a, b, choice, comment, kind, seed, context)
        except InvalidFeedback as error:
            # An unusable optional interpretation contributes no loss; keep the original remark.
            parse_error = str(error)
    accepted = eligible_constraints(comment, claims)
    missing = round_row["missing_attribute"]
    return {
        "raw": comment,
        "claims": claims,
        "parse_error": parse_error,
        "eligible": accepted,
        "mentions_missing": any(claim.get("attribute") == missing for claim in claims),
        "eligible_missing": any(claim["attribute"] == missing for claim in accepted),
    }


def run_fixed(tasks: list[Task], client, output: Path, *, seeds=SEEDS, profile_limit: int | None = None,
              test_limit: int | None = None) -> None:
    """Collect each decision once and replay it for all three reward objectives."""
    train = [task for task in tasks if task.split == "train"]
    test = [task for task in tasks if task.split == "test"][:test_limit]
    selected_profiles = profiles()[:profile_limit]
    feedback, predictions = [], []
    for seed in seeds:
        for profile in selected_profiles:
            profile_id = profile["id"]
            print(f"[fixed] seed={seed} profile={profile_id}", flush=True)
            tape = []
            for step, task in enumerate(train, start=1):
                round_seed = stable_seed("train", seed, profile_id, task.id)
                context = f"training seed={seed} profile={profile_id} round={step} task={task.id}"
                a_features, b_features, missing = training_pair(profile, round_seed)
                a, b = task.render(*a_features), task.render(*b_features)
                choice = client.choose(profile, task.question, a, b, round_seed, context)
                row = {
                    "seed": seed, "profile": profile_id, "round": step, "task": task.id,
                    "question": task.question, "a": a, "b": b, "a_features": list(a_features),
                    "b_features": list(b_features), "choice": choice, "missing_attribute": missing,
                    "comments": {},
                }
                if step in COMMENT_ROUNDS:
                    for kind in GROUPS[1:]:
                        row["comments"][kind] = _asks_record(kind, client, profile, task, row, round_seed, context)
                tape.append(row)
            feedback.extend(tape)
            weights = {
                (step, group): fit_reward(tape[:step], group)
                for step in CHECKPOINTS for group in GROUPS
            }
            for task in test:
                for kind in PAIR_TYPES:
                    pair_seed = stable_seed("test", seed, profile_id, task.id, kind)
                    a, b = test_pair(profile, kind, pair_seed)
                    context = f"fixed test seed={seed} profile={profile_id} task={task.id} pair={kind}"
                    choice = client.choose(profile, task.question, task.render(*a), task.render(*b), pair_seed, context)
                    for (step, group), w in weights.items():
                        prediction = predict_choice(w, a, b)
                        predictions.append({
                            "seed": seed, "profile": profile_id, "task": task.id, "pair": kind,
                            "round": step, "group": group, "a_features": list(a), "b_features": list(b),
                            "observed": choice, "prediction": prediction, "score": _score(prediction, choice),
                            "weights": w.tolist(),
                        })
    write_jsonl(output / "feedback.jsonl", feedback)
    write_jsonl(output / "fixed_predictions.jsonl", predictions)
    write_json(output / "fixed_run.json", {
        "task_fingerprint": task_fingerprint(tasks), "seeds": list(seeds),
        "profiles": [profile["id"] for profile in selected_profiles],
        "test_tasks": [task.id for task in test],
        "user_model": getattr(client, "user_model", None),
        "parser_model": getattr(client, "parser_model", None),
    })
    build_report(output)


def transcript(tape: list[dict]) -> str:
    """Give ICL the original comparison observations and raw edit comments."""
    rows = [{
        "question": row["question"], "A": row["a"], "B": row["b"],
        "selected": row["choice"], "comment": row["comments"].get("critique", {}).get("raw", ""),
    } for row in tape]
    return json.dumps(rows, ensure_ascii=False)


def _preferred_observed(tape: list[dict], observed: int) -> int | None:
    """The preserved attribute is inferred from actual choices, without a hidden label."""
    chosen = [row["a_features" if row["choice"] == "A" else "b_features"][observed] for row in tape]
    if 2 * sum(chosen) == len(chosen):
        return None
    return int(2 * sum(chosen) > len(chosen))


def _response_metrics(judgment: dict, side: str, observed: int, preserve: int | None, repair: int) -> dict:
    features = (judgment[f"lead_{side}"], judgment[f"example_{side}"])
    repaired = features[1 - observed] == repair
    kept = None if preserve is None else features[observed] == preserve
    return {
        "repair": repaired, "preserve": kept, "joint": None if kept is None else repaired and kept,
        "correct": judgment[f"correct_{side}"],
    }


def run_generate(
    tasks: list[Task], client, decoder, probe, fixed_dir: Path, output: Path, *,
    max_new_tokens: int = 128, top_k: int = 8, reward_weight: float = 1.0,
) -> None:
    manifest = json.loads((fixed_dir / "fixed_run.json").read_text(encoding="utf-8"))
    if manifest["task_fingerprint"] != task_fingerprint(tasks):
        raise ValueError("Fixed experiment and generated evaluation must use the same controlled task file")
    if getattr(client, "judge_model", None) == manifest["user_model"] and manifest["user_model"] is not None:
        raise ValueError("The judge model must differ from the fixed experiment's simulated user model")
    test_keys = set(manifest["test_tasks"])
    test = [task for task in tasks if task.split == "test" and task.id in test_keys]
    if not test or len(test) != len(test_keys):
        raise ValueError("Manifest does not contain a valid set of test tasks")
    tapes = defaultdict(list)
    for row in read_jsonl(fixed_dir / "feedback.jsonl"):
        tapes[row["seed"], row["profile"]].append(row)
    expected = {(seed, profile_id) for seed in manifest["seeds"] for profile_id in manifest["profiles"]}
    if set(tapes) != expected:
        raise ValueError("Feedback tape does not match the fixed experiment manifest")
    hidden = {profile["id"]: profile for profile in profiles()}
    expected_train = [task.id for task in tasks if task.split == "train"]
    generated, scored = [], []
    for (seed, profile_id), tape in sorted(tapes.items()):
        tape.sort(key=lambda row: row["round"])
        if [row["round"] for row in tape] != list(range(1, 17)):
            raise ValueError(f"Expected exactly 16 ordered feedback rounds for seed={seed}, profile={profile_id}")
        if [row["task"] for row in tape] != expected_train:
            raise ValueError(f"Feedback tape must contain only the ordered training tasks for profile={profile_id}")
        print(f"[generate] seed={seed} profile={profile_id}", flush=True)
        weights = {group: fit_reward(tape, group) for group in GROUPS}
        history = transcript(tape)
        profile = hidden[profile_id]
        observed = ATTRIBUTES.index(profile["priority"])
        preserve = _preferred_observed(tape, observed)
        repair = target_pair(profile)[1 - observed]
        for task in test:
            key = {"seed": seed, "profile": profile_id, "task": task.id}
            replies = {}
            # The generator sees the task plus learned weights or observed history only.
            for group in (*GROUPS, "prompt"):
                try:
                    replies[group] = decoder.generate(
                        task.question, weights=weights.get(group), probe=probe if group in GROUPS else None,
                        history=history if group == "prompt" else "", max_new_tokens=max_new_tokens,
                        top_k=top_k, reward_weight=reward_weight,
                    )
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Decoding failed for seed={seed} profile={profile_id} task={task.id} group={group}: {error}"
                    ) from error
                generated.append({
                    **key, "group": group, "response": replies[group],
                    "format_valid": extract_features(replies[group]) is not None,
                })
            for group in GROUPS:
                pair_seed = stable_seed("blind", seed, profile_id, task.id, group)
                left, right = orient(group, "prompt", pair_seed)
                # Give the judge a factual reference with stable formatting across profiles.
                reference = task.render(1, 1)
                context = f"generation seed={seed} profile={profile_id} task={task.id} group={group}"
                verdict = client.judge(profile, task.question, reference, replies[left], replies[right], pair_seed, context)
                side = "a" if left == group else "b"
                counterpart = "b" if side == "a" else "a"
                scored.append({
                    **key, "group": group, "method_side": side.upper(), "judgment": verdict,
                    "win_score": _score(verdict["winner"], side.upper()),
                    "metrics": _response_metrics(verdict, side, observed, preserve, repair),
                    "prompt_metrics": _response_metrics(verdict, counterpart, observed, preserve, repair),
                })
    write_jsonl(output / "generations.jsonl", generated)
    write_jsonl(output / "generation_scores.jsonl", scored)
    write_json(output / "generation_run.json", {
        "fixed_dir": str(fixed_dir.resolve()), "model": decoder.model_id,
        "judge_model": getattr(client, "judge_model", None), "max_new_tokens": max_new_tokens,
        "top_k": top_k, "reward_weight": reward_weight, "task_fingerprint": task_fingerprint(tasks),
    })
    build_report(output)


def _estimate(rows: list[tuple[str, float | bool | None]]) -> dict:
    """Bootstrap at the profile level, keeping seeds and prompts paired within it."""
    clusters = defaultdict(list)
    for profile_id, value in rows:
        if value is not None:
            clusters[profile_id].append(float(value))
    means = np.array([np.mean(clusters[key]) for key in sorted(clusters)])
    if not len(means):
        return {"observations": 0, "profiles": 0, "score_sum": 0.0, "mean": None, "interval_95": None}
    interval = None
    if len(means) > 1:
        rng = np.random.default_rng(17)
        boot = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
        interval = np.percentile(boot, [2.5, 97.5]).tolist()
    return {
        "observations": sum(map(len, clusters.values())), "profiles": len(means),
        "score_sum": sum(sum(value) for value in clusters.values()),
        "mean": float(means.mean()), "interval_95": interval,
    }


def _fixed_report(predictions: list[dict], feedback: list[dict]) -> dict:
    summaries = []
    for step in CHECKPOINTS:
        for kind in (*PAIR_TYPES, "all"):
            rows = [row for row in predictions if row["round"] == step and (kind == "all" or row["pair"] == kind)]
            choice_scores = {
                (row["seed"], row["profile"], row["task"], row["pair"]): row["score"]
                for row in rows if row["group"] == "choice"
            }
            for group in GROUPS:
                subset = [row for row in rows if row["group"] == group]
                paired = [(
                    row["profile"], row["score"] - choice_scores[row["seed"], row["profile"], row["task"], row["pair"]],
                ) for row in subset]
                summaries.append({
                    "round": step, "pair": kind, "group": group,
                    **_estimate([(row["profile"], row["score"]) for row in subset]),
                    "correct": sum(row["score"] == 1 for row in subset),
                    "ties": sum(row["score"] == 0.5 for row in subset),
                    "incorrect": sum(row["score"] == 0 for row in subset),
                    "delta_vs_choice": _estimate(paired),
                })
    budgets = {}
    for group in GROUPS[1:]:
        comments = [row["comments"][group] for row in feedback if group in row["comments"]]
        budgets[group] = {
            "requests": len(comments), "nonempty": sum(bool(item["raw"].strip()) for item in comments),
            "characters": sum(len(item["raw"]) for item in comments),
            "eligible": sum(bool(item["eligible"]) for item in comments),
            "mentions_missing": sum(item["mentions_missing"] for item in comments),
            "eligible_missing": sum(item["eligible_missing"] for item in comments),
            "parse_errors": sum(item.get("parse_error") is not None for item in comments),
        }
    return {"choices": len(feedback), "comments": budgets, "results": summaries}


def _generation_report(scores: list[dict], responses: list[dict]) -> list[dict]:
    summaries = []
    for group in GROUPS:
        rows = [row for row in scores if row["group"] == group]
        summary = {
            "group": group,
            "win_vs_prompt": _estimate([(row["profile"], row["win_score"]) for row in rows]),
            "wins": sum(row["win_score"] == 1 for row in rows),
            "ties": sum(row["win_score"] == 0.5 for row in rows),
            "losses": sum(row["win_score"] == 0 for row in rows),
            "format_valid": _estimate([(row["profile"], row["format_valid"]) for row in responses if row["group"] == group]),
            "prompt_format_valid": _estimate([(row["profile"], row["format_valid"]) for row in responses if row["group"] == "prompt"]),
        }
        for metric in ("repair", "preserve", "joint", "correct"):
            values = [(row["profile"], row["metrics"][metric]) for row in rows]
            prompt_values = [(row["profile"], row["prompt_metrics"][metric]) for row in rows]
            paired = [(p, int(value) - int(control)) for (p, value), (_, control) in zip(values, prompt_values)
                      if value is not None and control is not None]
            summary[metric] = {
                "method": _estimate(values), "prompt": _estimate(prompt_values), "paired_delta": _estimate(paired),
            }
        summaries.append(summary)
    return summaries


def _csv(path: Path, rows: list[dict]) -> None:
    if rows:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        write_text(path, buffer.getvalue())


def build_report(output: Path) -> list[Path]:
    """Read raw JSONL rather than contacting the simulator or loading a model."""
    written = []
    fixed = output / "fixed_predictions.jsonl"
    if fixed.exists():
        report = _fixed_report(read_jsonl(fixed), read_jsonl(output / "feedback.jsonl"))
        path = output / "fixed_metrics.json"
        write_json(path, report)
        _csv(output / "fixed_metrics.csv", [{key: value for key, value in row.items()
                                              if key not in {"delta_vs_choice"}} for row in report["results"]])
        written.append(path)
    scores = output / "generation_scores.jsonl"
    if scores.exists():
        report = _generation_report(read_jsonl(scores), read_jsonl(output / "generations.jsonl"))
        path = output / "generation_metrics.json"
        write_json(path, report)
        _csv(output / "generation_metrics.csv", [{
            "group": row["group"], "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
            "win_vs_prompt": row["win_vs_prompt"]["mean"],
            **{metric: row[metric]["method"]["mean"] for metric in ("repair", "preserve", "joint", "correct")},
            **{f"prompt_{metric}": row[metric]["prompt"]["mean"] for metric in ("repair", "preserve", "joint", "correct")},
        } for row in report])
        written.append(path)
    return written
