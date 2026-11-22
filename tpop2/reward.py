"""Two interpretable response attributes and a small Bradley–Terry reward head."""

from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ATTRIBUTES = ("lead", "example")
LABEL = re.compile(r"^\s*(?:\*\*)?(Answer|Explanation|Example):(?:\*\*)?\s*(.*)$", re.I)


def extract_features(response: str) -> tuple[int, int] | None:
    """Read the common output format, returning None when the structure is invalid."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in response.strip().splitlines():
        if not line.strip():
            continue
        match = LABEL.match(line)
        if match:
            current = match[1].lower()
            if current in sections:
                return None
            sections[current] = [match[2]]
        elif current is None:
            return None
        else:
            sections[current].append(line)

    if not {"answer", "explanation"}.issubset(sections):
        return None
    if any(not " ".join(lines).strip() for lines in sections.values()):
        return None
    return int(next(iter(sections)) == "answer"), int("example" in sections)


def choice_difference(a, b, choice: str) -> np.ndarray:
    """Orient a pair toward its winner so every likelihood target is positive."""
    if choice not in {"A", "B"}:
        raise ValueError(f"Invalid choice {choice!r}; expected A or B")
    a_features, b_features = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_features.shape != (2,) or b_features.shape != (2,):
        raise ValueError("Expected two features per answer")
    diff = a_features - b_features
    return diff if choice == "A" else -diff


def eligible_constraints(comment: str, claims: list[dict]) -> list[dict]:
    """Keep supported, future-facing attribute preferences, once per attribute."""
    candidates: dict[str, list[dict]] = {}
    for claim in claims:
        attr = claim.get("attribute")
        direction = claim.get("direction")
        evidence = claim.get("evidence")
        if (
            attr in ATTRIBUTES
            and type(direction) is int and direction in {-1, 1}
            and claim.get("scope") == "future"
            and isinstance(evidence, str) and evidence.strip() and evidence in comment
        ):
            candidates.setdefault(attr, []).append(claim)

    # Conflicting directions for the same attribute do not define a usable edit.
    return [
        candidates[attr][0] for attr in ATTRIBUTES if attr in candidates
        and len({claim["direction"] for claim in candidates[attr]}) == 1
    ]


def loss_and_gradient(weights, differences, importance, l2: float) -> tuple[float, np.ndarray]:
    """Regularized negative log-likelihood with a stable log-sigmoid."""
    margin = differences @ weights
    loss = importance @ np.logaddexp(0.0, -margin) + l2 * (weights @ weights)
    gradient = differences.T @ (-importance * expit(-margin)) + 2 * l2 * weights
    return float(loss), gradient


def fit_reward(records: list[dict], group: str, *, l2: float = 1.0, text_weight: float = 0.5) -> np.ndarray:
    """Replay the same choice observations, adding one kind of text when enabled."""
    if group not in {"choice", "reason", "critique"}:
        raise ValueError(f"Unknown reward group {group!r}")
    if not np.isfinite(l2) or not np.isfinite(text_weight) or l2 <= 0 or text_weight < 0:
        raise ValueError("Expected finite l2 > 0 and text_weight >= 0")
    differences, importance = [], []
    for record in records:
        differences.append(choice_difference(record["a_features"], record["b_features"], record["choice"]))
        importance.append(1.0)
        text = record["comments"].get(group)
        if group == "choice" or not text or text_weight == 0:
            continue
        constraints = eligible_constraints(text["raw"], text["claims"])
        for constraint in constraints:
            # A hypothetical edit changes only the named attribute, not the A/B label.
            delta = np.zeros(2)
            delta[ATTRIBUTES.index(constraint["attribute"])] = constraint["direction"]
            differences.append(delta)
            importance.append(text_weight / len(constraints))

    if not differences:
        return np.zeros(2)
    result = minimize(
        loss_and_gradient,
        np.zeros(2),
        args=(np.asarray(differences), np.asarray(importance), l2),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(f"Could not fit reward weights: {result.message}")
    return result.x


def predict_choice(weights, a, b) -> str:
    """Expose exact ties rather than arbitrarily favoring option A."""
    margin = float(np.asarray(weights) @ choice_difference(a, b, "A"))
    if abs(margin) <= 1e-9:
        return "tie"
    return "A" if margin > 0 else "B"
