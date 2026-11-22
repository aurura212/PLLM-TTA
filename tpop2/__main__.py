"""CLI for data collection, a shared probe, generation, and offline reporting."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .data import load_tasks, probe_questions
from .decode import LocalDecoder, PrefixProbe
from .experiment import SEEDS, build_report, run_fixed, run_generate, write_json
from .feedback import FeedbackClient
from .reward import ATTRIBUTES

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
TASKS = ROOT / "data" / "controlled_tasks.jsonl"
HELPSTEER = ROOT.parent / "T-POP-main" / "data" / "HelpSteer_train.json"


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite, non-negative number")
    return number


def _task_file(path: Path):
    tasks = load_tasks(path)
    if Counter(task.split for task in tasks) != {"train": 16, "dev": 8, "test": 16}:
        raise ValueError("The controlled task file must contain exactly 16 train, 8 dev, and 16 test tasks")
    return tasks


def _probe(args, tasks) -> None:
    train, valid = probe_questions(args.helpsteer, tasks, seed=args.seed)
    model = LocalDecoder(args.model, args.device)
    print(f"[probe] Sampling {len(train)} training prompts", flush=True)
    x, y, train_sources = model.probe_examples(train, args.seed, args.max_new_tokens)
    print(f"[probe] Sampling {len(valid)} validation prompts", flush=True)
    vx, vy, valid_sources = model.probe_examples(valid, args.seed + 1, args.max_new_tokens)
    if not len(x) or not len(vx):
        raise ValueError(
            "The local model produced no parseable responses for one probe split. "
            "Use an instruction-tuned checkpoint or increase --max-new-tokens."
        )
    probe = PrefixProbe.fit(x, y)
    predicted = probe.predict(vx)
    train_mean = y.mean(axis=0)
    mse = ((predicted - vy) ** 2).mean(axis=0)
    baseline = ((train_mean - vy) ** 2).mean(axis=0)
    coverage = {
        name: {
            "train_positive_prefixes": int(y[:, i].sum()),
            "validation_positive_prefixes": int(vy[:, i].sum()),
            "validation_mse": float(mse[i]), "training_mean_baseline_mse": float(baseline[i]),
        } for i, name in enumerate(ATTRIBUTES)
    }
    for i, name in enumerate(ATTRIBUTES):
        if len(np.unique(y[:, i])) < 2 or mse[i] >= baseline[i]:
            print(f"[probe] Warning: {name} is constant in training or fails to beat the validation mean baseline.",
                  flush=True)
    probe.save(args.out_dir / "probe.npz", model.model_id)
    write_json(args.out_dir / "probe_metrics.json", {
        "model": model.model_id, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "source": str(args.helpsteer.resolve()), "train_prompts": train, "validation_prompts": valid,
        "train_prefixes": len(x), "validation_prefixes": len(vx),
        "train_covered_prompts": len(set(train_sources)), "validation_covered_prompts": len(set(valid_sources)),
        "attributes": coverage,
    })


def _model_arguments(command) -> None:
    command.add_argument("--model", type=Path, required=True, help="Local HF causal LM directory")
    command.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    command.add_argument("--max-new-tokens", type=_positive, default=128)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    fixed = commands.add_parser("fixed", help="Collect feedback and evaluate fixed candidates")
    fixed.add_argument("--tasks", type=Path, default=TASKS)
    fixed.add_argument("--out-dir", type=Path, default=OUTPUT / "fixed")
    fixed.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    fixed.add_argument("--profile-limit", type=int, choices=range(1, 9), default=8)
    fixed.add_argument("--test-limit", type=int, choices=range(1, 17), default=16)

    probe = commands.add_parser("probe", help="Fit the local frozen model's shared prefix probe")
    probe.add_argument("--tasks", type=Path, default=TASKS)
    probe.add_argument("--helpsteer", type=Path, default=HELPSTEER)
    probe.add_argument("--out-dir", type=Path, default=OUTPUT / "probe")
    probe.add_argument("--seed", type=int, default=17)
    _model_arguments(probe)

    generate = commands.add_parser("generate", help="Replay feedback and blind-judge four generation groups")
    generate.add_argument("--tasks", type=Path, default=TASKS)
    generate.add_argument("--fixed-dir", type=Path, default=OUTPUT / "fixed")
    generate.add_argument("--probe-file", type=Path, default=OUTPUT / "probe" / "probe.npz")
    generate.add_argument("--out-dir", type=Path, default=OUTPUT / "generation")
    generate.add_argument("--top-k", type=_positive, default=8)
    generate.add_argument("--reward-weight", type=_nonnegative, default=1.0)
    _model_arguments(generate)

    report = commands.add_parser("report", help="Rebuild tables from saved JSONL without model or API access")
    report.add_argument("--run-dir", type=Path, nargs="+", default=[OUTPUT / "fixed", OUTPUT / "generation"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.stage == "report":
            reports = [path for directory in args.run_dir for path in build_report(directory)]
            if not reports:
                raise ValueError("No raw experiment results in the selected directories")
            for path in reports:
                print(path)
            return 0

        tasks = _task_file(args.tasks)
        if args.stage == "probe":
            _probe(args, tasks)
        elif args.stage == "fixed":
            if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
                raise ValueError("Use distinct non-negative experiment seeds")
            client = FeedbackClient.from_env(args.out_dir / "api_cache")
            run_fixed(tasks, client, args.out_dir, seeds=tuple(args.seeds),
                      profile_limit=args.profile_limit, test_limit=args.test_limit)
        else:
            client = FeedbackClient.from_env(args.out_dir / "api_cache")
            model = LocalDecoder(args.model, args.device)
            prefix_probe = PrefixProbe.load(args.probe_file, model.model_id)
            run_generate(tasks, client, model, prefix_probe, args.fixed_dir, args.out_dir,
                         max_new_tokens=args.max_new_tokens, top_k=args.top_k, reward_weight=args.reward_weight)
        print(f"[{args.stage}] Outputs in {args.out_dir.resolve()}")
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f"tpop2 {args.stage}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
