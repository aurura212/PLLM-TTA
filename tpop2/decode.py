"""Local causal-LM decoding and a shared, offline-trained prefix outcome probe."""

from pathlib import Path
import hashlib
import inspect

import numpy as np


FEATURE_NAMES = ("lead", "example")
FORMAT_INSTRUCTION = (
    "Use the headings 'Answer:' and 'Explanation:' exactly once each, at the start "
    "of separate lines. Put these two sections first, in whichever order suits the "
    "response. You may finish with an 'Example:' section when useful. Write content under each "
    "heading you use."
)


def _matrix(values, name, columns=None):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or (columns is not None and matrix.shape[1] != columns):
        raise ValueError(f"{name} must be a matrix with the expected number of columns")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def choose_token(token_ids, base_probs, candidate_features=None, weights=None, reward_weight=1.0):
    """Return a token ID. Base probabilities are on the full-vocabulary scale."""
    ids = np.asarray(token_ids)
    probs = np.asarray(base_probs, dtype=np.float64)
    if ids.ndim != 1 or not len(ids) or probs.shape != ids.shape:
        raise ValueError("Require a nonempty vector of token IDs and one probability per ID")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Token IDs must be non-negative integers")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid base probability")
    if not np.isfinite(reward_weight) or reward_weight < 0:
        raise ValueError("reward_weight must be finite and non-negative")
    scores = probs
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(FEATURE_NAMES),) or not np.isfinite(w).all():
            raise ValueError("Require two finite preference weights (lead, example)")
        if np.any(w) and reward_weight > 0:
            features = _matrix(candidate_features, "candidate features", len(w))
            if len(features) != len(ids):
                raise ValueError("One feature vector is required per candidate")
            z = np.clip(features @ w, -60.0, 60.0)
            scores = probs + reward_weight / (1.0 + np.exp(-z))
    return int(ids[np.argmax(scores)])


class PrefixProbe:
    """Ridge regression from frozen prefix states to two terminal attributes."""

    def __init__(self, coef, intercept):
        self.coef = _matrix(coef, "coefficients", len(FEATURE_NAMES))
        self.intercept = np.asarray(intercept, dtype=np.float64)
        if not self.coef.shape[0] or self.intercept.shape != (len(FEATURE_NAMES),):
            raise ValueError("Invalid probe dimensions")
        if not np.isfinite(self.intercept).all():
            raise ValueError("Non-finite probe intercept")

    @classmethod
    def fit(cls, x, y):
        x = _matrix(x, "prefix states")
        y = _matrix(y, "terminal labels", len(FEATURE_NAMES))
        if not x.size or len(x) != len(y) or np.any((y < 0) | (y > 1)):
            raise ValueError("Need aligned, nonempty prefix states and labels in [0,1]")
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - x_mean, y - y_mean
        if len(x) < x.shape[1]:
            gram = xc @ xc.T
            gram.flat[::len(gram) + 1] += 1.0
            coef = xc.T @ np.linalg.solve(gram, yc)
        else:
            gram = xc.T @ xc
            gram.flat[::len(gram) + 1] += 1.0
            coef = np.linalg.solve(gram, xc.T @ yc)
        return cls(coef, y_mean - x_mean @ coef)

    def predict(self, x):
        x = _matrix(x, "prefix states", self.coef.shape[0])
        return np.clip(x @ self.coef + self.intercept, 0.0, 1.0)

    def save(self, path, model_id):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty string")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, version=1, model_id=model_id, names=FEATURE_NAMES,
                                coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path, model_id):
        with np.load(path, allow_pickle=False) as data:
            if data["version"].item() != 1 or tuple(data["names"]) != FEATURE_NAMES:
                raise ValueError("Incompatible prefix probe format or feature ordering")
            if data["model_id"].item() != model_id:
                raise ValueError("Probe was trained with a different model")
            return cls(data["coef"], data["intercept"])


class LocalDecoder:
    def __init__(self, model_path, device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path).expanduser().resolve(strict=True)
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype="auto"
        ).to(self.device).eval()
        self.width = self.model.config.hidden_size
        self.context_size = getattr(self.model.config, "max_position_embeddings", None)
        fingerprint = hashlib.sha256((path / "config.json").read_bytes())
        # Detect in-place checkpoint/tokenizer replacements without reading GB of
        # weights a second time. Model directories are expected to be immutable.
        for file in sorted(path.iterdir()):
            if file.suffix in {".safetensors", ".bin", ".model"} or file.name.startswith("tokenizer"):
                info = file.stat()
                fingerprint.update(f"{file.name}:{info.st_size}:{info.st_mtime_ns}\n".encode())
        self.model_id = f"{path.as_posix()}|{fingerprint.hexdigest()[:16]}|hidden={self.width}"
        stops = self.model.generation_config.eos_token_id
        self.eos_ids = set(stops if isinstance(stops, (list, tuple)) else [stops])
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids.discard(None)
        self.logit_kwargs = {}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            self.logit_kwargs["logits_to_keep"] = 1

    def _prompt(self, question, history=""):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        prompt = f"{FORMAT_INSTRUCTION}\n\n"
        if history:
            prompt += f"User feedback from earlier interactions:\n{history}\n\n"
        prompt += f"Question:\n{question.strip()}"
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            add_special_tokens = False
        else:
            prompt += "\n\nResponse:\n"
            add_special_tokens = True
        return self.tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=add_special_tokens).input_ids.to(self.device)

    def _budget(self, ids, max_new_tokens):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.context_size is not None and ids.shape[1] + max_new_tokens > self.context_size:
            raise ValueError(
                f"Prompt ({ids.shape[1]} tokens) plus generation budget ({max_new_tokens}) "
                f"exceeds the context window ({self.context_size}). Choose a longer-context model "
                "or use a smaller --max-new-tokens for every group."
            )
        return max_new_tokens

    def _text(self, response_ids):
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def generate(self, question, weights=None, probe=None, history="", max_new_tokens=128,
                 top_k=8, reward_weight=1.0):
        from .reward import extract_features

        torch = self.torch
        if top_k < 1:
            raise ValueError("top_k must be positive")
        # Validate arguments even when all weights are zero.
        choose_token([0], [1.0], [[0.0, 0.0]], weights, reward_weight)
        guided = weights is not None and np.any(weights) and reward_weight > 0
        if guided and probe is None:
            raise ValueError("Guided generation requires a fitted prefix probe")
        ids = self._prompt(question, history)
        response = []
        with torch.inference_mode():
            budget = self._budget(ids, max_new_tokens)
            out = self.model(input_ids=ids, use_cache=True, **self.logit_kwargs)
            for step in range(budget):
                logits = out.logits[0, -1].float()
                cache = getattr(out, "past_key_values", None)
                if not guided:
                    token = int(logits.argmax())
                    response.append(token)
                    if token in self.eos_ids or step + 1 == budget:
                        break
                    token_ids = ids.new_tensor([[token]])
                    ids = torch.cat((ids, token_ids), dim=1)
                    out = self.model(input_ids=token_ids if cache is not None else ids,
                                     past_key_values=cache, use_cache=True, **self.logit_kwargs)
                    continue

                probs, candidates = torch.topk(logits.softmax(-1), min(top_k, logits.numel()))
                ids_cpu = candidates.tolist()
                # Score h+v for every v in one model call. Standard HF dynamic caches
                # support branching and selecting the winning row. Other models use
                # full prefixes, leaving the mathematics identical.
                branch_cache = hasattr(cache, "batch_repeat_interleave") and hasattr(cache, "batch_select_indices")
                if branch_cache:
                    cache.batch_repeat_interleave(len(ids_cpu))
                    batch = candidates[:, None]
                else:
                    cache = None
                    batch = torch.cat((ids.expand(len(ids_cpu), -1), candidates[:, None]), dim=1)
                branches = self.model(input_ids=batch, past_key_values=cache,
                                      use_cache=branch_cache, output_hidden_states=True, **self.logit_kwargs)
                hidden = branches.hidden_states[-1][:, -1, :].float().cpu().numpy()
                features = probe.predict(hidden)
                for i, token_id in enumerate(ids_cpu):
                    if token_id in self.eos_ids or step + 1 == budget:
                        terminal = extract_features(self._text(response + [token_id]))
                        if terminal is not None:
                            features[i] = terminal
                token = choose_token(ids_cpu, probs.cpu().numpy(), features, weights, reward_weight)
                winner = ids_cpu.index(token)
                response.append(token)
                if token in self.eos_ids or step + 1 == budget:
                    break
                ids = torch.cat((ids, candidates[winner].view(1, 1)), dim=1)
                if branch_cache:
                    branches.past_key_values.batch_select_indices(candidates.new_tensor([winner]))
                branches.logits = branches.logits[winner:winner+1, -1:, :]
                branches.hidden_states = None
                out = branches
        return self._text(response)

    def probe_examples(self, questions, seed, max_new_tokens=128):
        """Sample four independent continuations per question, retaining up to five prefixes per valid text."""
        from .reward import extract_features

        torch = self.torch
        states, labels, sources = [], [], []
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            for question in questions:
                ids = self._prompt(question)
                length = ids.shape[1]
                sequences = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                    max_new_tokens=self._budget(ids, max_new_tokens),
                    num_beams=1, num_return_sequences=4, return_dict_in_generate=False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                    else min(self.eos_ids, default=0),
                    eos_token_id=sorted(self.eos_ids) or None,
                )
                for seq in sequences:
                    response = seq[length:].tolist()
                    end = next((i for i, v in enumerate(response) if v in self.eos_ids), len(response))
                    terminal = extract_features(self._text(response[:end]))
                    if terminal is None or not end:
                        continue
                    # Position n is the post-token state for a prefix of n new tokens.
                    counts = sorted({1, max(1, end // 4), max(1, end // 2),
                                     max(1, 3 * end // 4), end})
                    result = self.model(input_ids=seq[:length+end].unsqueeze(0),
                                        use_cache=False, output_hidden_states=True, **self.logit_kwargs)
                    positions = [length + n - 1 for n in counts]
                    states.append(result.hidden_states[-1][0, positions].float().cpu().numpy())
                    labels.extend([terminal] * len(counts))
                    sources.extend([question] * len(counts))
        x = np.concatenate(states) if states else np.empty((0, self.width), dtype=np.float32)
        return x, np.asarray(labels, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)), sources
