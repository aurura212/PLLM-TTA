"""Controlled explanation tasks, user profiles, and disjoint probe prompts."""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import unicodedata


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    question: str
    answer: str
    explanation: str
    example: str

    def render(self, lead: int, example: int) -> str:
        """Render one of four styles without changing the answer or reasoning."""
        if lead not in (0, 1) or example not in (0, 1):
            raise ValueError("lead and example must each be 0 or 1")
        conclusion = f"Answer: {self.answer}"
        body = [f"Explanation: {self.explanation}"]
        if example:
            body.append(f"Example: {self.example}")
        if lead:
            return "\n".join([conclusion, *body])
        return "\n".join([*body, conclusion])


def normalize_question(text: str) -> str:
    """Create a Unicode-, case-, and whitespace-normalized question key."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def stable_seed(*parts) -> int:
    """Derive a reproducible 32-bit seed from a sequence of JSON primitives."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def load_tasks(path) -> list[Task]:
    """Read the checked-in JSONL fixture, rejecting unusable or duplicate tasks."""
    tasks = []
    seen_ids = set()
    seen_questions = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                task = Task(**json.loads(line))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid task at {path}:{number}: {error}") from error
            if any(not isinstance(value, str) or not value.strip() for value in vars(task).values()):
                raise ValueError(f"Empty or non-text task field at {path}:{number}")
            if task.split not in {"train", "dev", "test"}:
                raise ValueError(f"Invalid split at {path}:{number}: {task.split}")
            key = normalize_question(task.question)
            if task.id in seen_ids or key in seen_questions:
                raise ValueError(f"Duplicate task id or question at {path}:{number}")
            seen_ids.add(task.id)
            seen_questions.add(key)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


def probe_questions(path, excluded, seed=17) -> tuple[list[str], list[str]]:
    """Sample 72 training and 24 validation questions from a prompt-only JSON file."""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list of prompts in {path}")
    blocked = {
        normalize_question(item.question if isinstance(item, Task) else item)
        for item in excluded
    }
    unique = {}
    for row in rows:
        question = row.get("question") if isinstance(row, dict) else row
        if not isinstance(question, str) or not question.strip():
            continue
        key = normalize_question(question)
        if key not in blocked:
            unique.setdefault(key, question.strip())
    if len(unique) < 96:
        raise ValueError(f"Need 96 non-overlapping probe prompts; found {len(unique)} in {path}")
    keys = random.Random(seed).sample(sorted(unique), 96)
    prompts = [unique[key] for key in keys]
    return prompts[:72], prompts[72:]


def profiles() -> list[dict]:
    """Return the complete 2 x 2 x 2 design (two preferences and a priority)."""
    users = []
    for lead, example, priority in itertools.product((0, 1), (0, 1), ("lead", "example")):
        placement = "first" if lead else "last"
        example_preference = "Include a concrete example." if example else "Omit the example."
        priority_text = "answer placement" if priority == "lead" else "my preference about examples"
        users.append({
            "id": f"lead{lead}_example{example}_{priority}",
            "lead": lead,
            "example": example,
            "priority": priority,
            "description": (
                f"Put the answer {placement}. {example_preference} "
                f"If these preferences conflict, prioritize {priority_text}."
            ),
        })
    return users
