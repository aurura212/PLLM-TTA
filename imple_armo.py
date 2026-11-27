import os
import json
import argparse
import math
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from judger import PREFERENCE_ATTRIBUTES
import numpy as np
import copy
import random
from tqdm import tqdm
from typing import List, Dict
import time
from torch.utils.data import DataLoader, Dataset

from model import MABRewardGenerator
from judger import PreferenceCollector
from rm_eval import (
    PersonalizationRMEvaluator, Question, load_questions, make_score_rows, summarize_scores, write_jsonl,
)

REVISION_INSTRUCTION = (
    "Rewrite the selected answer to the original question so it addresses the user's feedback. "
    "Preserve useful and correct details unrelated to the requested change, and avoid unverified facts. "
    "Return only the complete revised answer, without explaining what you changed."
)

# ---------------- Training Components ----------------
class PreferenceDataset(Dataset):
    def __init__(self, path: str, include_critiques: bool = False):
        self.samples = []
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if all(k in rec for k in ("y1", "y2", "label")):
                        sample = (rec["y1"], rec["y2"], rec["label"])
                        if include_critiques:
                            critique = rec.get("critique", {})
                            revision = ""
                            if critique.get("status") == "usable" and critique.get("revision_source") == "base_llm":
                                revision = critique.get("revised_y", "")
                            sample = (*sample, revision)
                        self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pairwise_loss(s1: torch.Tensor, s2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    margin = s1.view(-1) - s2.view(-1)
    # Upstream collector and win-rate evaluator both encode A=0 and B=1.
    # BCE on (s1-s2) needs a positive target when A won.
    y = 1.0 - label.view(-1).float()
    return F.binary_cross_entropy_with_logits(margin, y)


def critique_loss(generator, s1, s2, labels, revisions):
    """Compare assistant revisions with their chosen originals in the same batch."""
    indices = [idx for idx, revision in enumerate(revisions) if revision]
    if not indices:
        return None
    original = torch.where(labels.view(-1) == 0, s1.view(-1), s2.view(-1))[indices]
    embs = generator.get_embedding([revisions[idx] for idx in indices]).to(original.device)
    revised = generator.reward_model(embs).view(-1)
    # Divide by the original batch size: missing feedback contributes zero.
    # Keep the number of training examples and optimizer steps unchanged.
    return F.softplus(original - revised).sum() / labels.numel()

@torch.inference_mode()
def generate_revision(generator, query, answer, comment, max_new_tokens):
    """Use the frozen assistant to answer again in light of the feedback."""
    item = json.dumps({"question": query, "selected_answer": answer, "user_feedback": comment}, ensure_ascii=False)
    tokenizer = generator.tokenizer
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{REVISION_INSTRUCTION}\n\n{item}"}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = f"{REVISION_INSTRUCTION}\n\n{item}\n\nRevised answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=not has_chat_template).to(generator.llm_device)
    prompt_length = inputs["input_ids"].shape[-1]
    context_length = getattr(generator.LLM.config, "max_position_embeddings", None)
    if context_length is not None:
        max_new_tokens = min(max_new_tokens, context_length - prompt_length)
    if max_new_tokens < 1:
        raise ValueError("No context space available for the assistant's revision")
    # The separate greedy call leaves the original dual-arm generation and
    # its exploration state untouched.
    ids = generator.LLM.generate(
        **inputs, do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=max_new_tokens, return_dict_in_generate=False,
    )
    return tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True).strip()


def add_assistant_revision(generator, critique, enhanced_query, selected_answer, max_new_tokens):
    """Keep the binary choice even when revision generation fails or adds no signal."""
    try:
        revised = generate_revision(generator, enhanced_query, selected_answer, critique["comment"], max_new_tokens)
    except Exception as error:
        critique["status"] = "revision_failed"
        print(f"[WARNING] Assistant revision failed ({type(error).__name__}): {error}")
        return
    if not revised or revised == selected_answer.strip():
        critique["status"] = "revision_unusable"
        return
    critique.update(status="usable", revision_source="base_llm", revised_answer=revised,
                    revised_y=f"{enhanced_query} {revised}")


def reinit_reward_model(model: nn.Module, init_state_dict: dict):
    model.load_state_dict(copy.deepcopy(init_state_dict))


def construct_prompt_with_preference(query: str, attribute: str) -> str:
    instruction = PREFERENCE_ATTRIBUTES.get(attribute, "")
    if instruction:
        return f"{query} {instruction}"
    else:
        return query

def generate_with_fixed_weight(generator: MABRewardGenerator, query: str,
                              max_new_tokens: int, pre_screen_beam_width: int, reward_weight: float,
                              nu_mab: float = 1.0, lambda_mab: float = 1.0, attribute: str = "creative") -> str:
    # Construct enhanced prompt with preference instruction
    enhanced_prompt = construct_prompt_with_preference(query, attribute)

    try:
        ids1, _ = generator.generate(
            prompt=enhanced_prompt,
            weight=reward_weight,
            max_new_tokens=max_new_tokens,
            pre_screen_beam_width=pre_screen_beam_width,
            nu_mab=nu_mab,
            lambda_mab=lambda_mab
        )

        raw_response = generator.tokenizer.decode(ids1, skip_special_tokens=True)
        # Strip prompt echo (including preference instruction)
        if raw_response.startswith(enhanced_prompt):
            response = raw_response[len(enhanced_prompt):].lstrip()
        else:
            response = raw_response

        return response
    except Exception:
        return ""


def configured_eval_parallel(cfg) -> int:
    """How many evaluation questions are answered at the same time."""
    value = int(getattr(cfg, "eval_parallel", 1))
    if value < 1:
        raise ValueError("--eval_parallel must be a positive integer")
    return value


RESUME_STATE = "resume_state.json"
TRAINING_QUERIES = "training_queries.json"


def _atomic_json(path: str, payload) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_resume_state(output_dir: str) -> dict:
    path = os.path.join(output_dir, RESUME_STATE)
    if not os.path.isfile(path):
        return {"completed_queries": [], "phase": "train"}
    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    state["completed_queries"] = [int(index) for index in state.get("completed_queries", [])]
    state["phase"] = state.get("phase", "train")
    return state


def save_resume_state(output_dir: str, state: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _atomic_json(os.path.join(output_dir, RESUME_STATE), state)


def mark_query_done(output_dir: str, state: dict, index: int) -> None:
    done = set(state["completed_queries"])
    done.add(index)
    state["completed_queries"] = sorted(done)
    save_resume_state(output_dir, state)


def saved_query_indexes(pref_path: str) -> set:
    found = set()
    if not os.path.isfile(pref_path):
        return found
    with open(pref_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "query_index" in record:
                found.add(int(record["query_index"]))
    return found


def load_saved_responses(path: str) -> Dict:
    answers = {}
    if not os.path.isfile(path):
        return answers
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("ours")
            if "question_id" in row and isinstance(text, str) and text.strip():
                answers[row["question_id"]] = text
    return answers


def result_has_score(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        return any(line.startswith("ArmoRM Score:") for line in handle)


def _response_from_ids(generator: MABRewardGenerator, ids: torch.Tensor, prompt: str) -> str:
    raw_response = generator.tokenizer.decode(ids, skip_special_tokens=True)
    if raw_response.startswith(prompt):
        return raw_response[len(prompt):].lstrip()
    return raw_response


def generate_with_fixed_weight_batch(generator: MABRewardGenerator, queries: List[str],
                                     max_new_tokens: int, pre_screen_beam_width: int, reward_weight: float,
                                     nu_mab: float = 1.0, lambda_mab: float = 1.0, attribute: str = "creative") -> List[str]:
    """Answer several evaluation questions in one decoding loop."""
    prompts = [construct_prompt_with_preference(query, attribute) for query in queries]
    try:
        pairs = generator.generate_batch(
            prompts,
            weight=reward_weight,
            max_new_tokens=max_new_tokens,
            pre_screen_beam_width=pre_screen_beam_width,
            nu_mab=nu_mab,
            lambda_mab=lambda_mab,
        )
    except Exception:
        return [""] * len(queries)
    if len(pairs) != len(prompts):
        return [""] * len(queries)

    responses = []
    for prompt, pair in zip(prompts, pairs):
        if not pair:
            responses.append("")
            continue
        try:
            responses.append(_response_from_ids(generator, pair[0], prompt))
        except Exception:
            responses.append("")
    return responses

# Baseline generation functions removed - not needed for ArmoRM evaluation

def train_reward(generator: MABRewardGenerator, judge: PreferenceCollector,
                 attribute: str, training_queries: List[str], cfg) -> str:

    init_model_weights = copy.deepcopy(generator.reward_model.state_dict())

    pref_dir = os.path.join(cfg.output_dir, "preferences")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(pref_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    pref_path = os.path.join(pref_dir, "preferences.jsonl")
    use_critiques = cfg.feedback_mode == "critique" and cfg.critique_weight > 0
    state = load_resume_state(cfg.output_dir)
    completed = set(state["completed_queries"]) | saved_query_indexes(pref_path)
    state["completed_queries"] = sorted(completed)
    final_ckpt = os.path.join(ckpt_dir, "final_model.pt")
    if completed >= set(range(len(training_queries))) and os.path.isfile(final_ckpt):
        print(f"   Training already finished ({len(training_queries)} queries).")
        return final_ckpt

    def fit_preferences(dataset: PreferenceDataset, label: str) -> int:
        """Refit the original reward weights on one snapshot of collected feedback."""
        reinit_reward_model(generator.reward_model, init_model_weights)
        sample_count = len(dataset)
        if sample_count == 0:
            return 0

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0
        )
        wd = 1.0 / (sample_count + 50)
        optimizer = torch.optim.AdamW(
            generator.reward_model.parameters(),
            lr=cfg.lr,
            weight_decay=wd
        )
        generator.reward_model.train()
        print(f"   {label}: wd={wd:.4f}")

        for ep in range(cfg.epochs_per_query):
            total_loss = 0.0
            batch_count = 0

            for batch in loader:
                y1, y2, lab = batch[:3]
                lab_t = lab.to(cfg.reward_model_device).float() \
                        if isinstance(lab, torch.Tensor) else torch.tensor(lab, device=cfg.reward_model_device).float()

                emb1 = generator.get_embedding(list(y1)).to(cfg.reward_model_device)
                emb2 = generator.get_embedding(list(y2)).to(cfg.reward_model_device)
                s1 = generator.reward_model(emb1)
                s2 = generator.reward_model(emb2)

                # Compute pairwise preference loss
                loss = pairwise_loss(s1, s2, lab_t)
                if use_critiques:
                    extra_loss = critique_loss(generator, s1, s2, lab_t, batch[3])
                    if extra_loss is not None:
                        loss = loss + cfg.critique_weight * extra_loss
                    del extra_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

                # Clean up intermediate tensors to prevent memory accumulation
                del emb1, emb2, s1, s2, loss, lab_t

                # Clear gradients and GPU cache every few batches to prevent accumulation
                if batch_count % 5 == 0:  # Every 5 batches
                    torch.cuda.empty_cache()
                    # Also clear BLLM internal cache if it exists
                    if hasattr(generator.BLLM, 'past_key_values'):
                        generator.BLLM.past_key_values = None

            # Print loss for every epoch
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            print(f"     Epoch {ep+1:3d}/{cfg.epochs_per_query}: loss = {avg_loss:.6f}")

            # Clear GPU cache after each epoch to prevent accumulation
            torch.cuda.empty_cache()

        # Clear optimizer and dataset after training to free memory
        del optimizer, dataset, loader
        torch.cuda.empty_cache()
        return sample_count

    fitted = 0
    for idx, q in enumerate(tqdm(training_queries, desc=f"Training")):
        if idx in completed:
            continue

        if idx >= 1:
            fitted = fit_preferences(
                PreferenceDataset(pref_path, include_critiques=use_critiques),
                f"Query {idx + 1}/{len(training_queries)}",
            )

        # 3. Generate two responses using current RM with specified weight
        enhanced_prompt = construct_prompt_with_preference(q, cfg.attribute)

        try:
            ids1, ids2 = generator.generate(
                prompt=enhanced_prompt,
                weight=cfg.reward_weight,
                max_new_tokens=cfg.max_new_tokens,
                pre_screen_beam_width=cfg.pre_screen_beam_width,
                nu_mab=cfg.nu_mab,
                lambda_mab=cfg.lambda_mab
            )
        except Exception:
            mark_query_done(cfg.output_dir, state, idx)
            completed.add(idx)
            continue

        # Decode & strip prompt echo
        raw1 = generator.tokenizer.decode(ids1, skip_special_tokens=True)
        raw2 = generator.tokenizer.decode(ids2, skip_special_tokens=True)

        # Strip the enhanced prompt (query + preference instruction) from the response
        response_1 = raw1[len(enhanced_prompt):].lstrip() if raw1.startswith(enhanced_prompt) else raw1
        response_2 = raw2[len(enhanced_prompt):].lstrip() if raw2.startswith(enhanced_prompt) else raw2

        # Clear generation-related variables and GPU cache
        del ids1, ids2, raw1, raw2
        torch.cuda.empty_cache()

        # 4. Collect preference data
        try:
            pref = judge.collect_single({
                "attribute": attribute,
                "query": q,  # Use original query for API prompt (avoid redundancy)
                "enhanced_query": enhanced_prompt,  # Pass enhanced_prompt for training data
                "response_1": response_1,
                "response_2": response_2,
            })
            if pref:
                critique = pref.get("critique")
                if use_critiques and critique and critique["comment"]:
                    selected_answer = response_1 if pref["label"] == 0 else response_2
                    add_assistant_revision(generator, critique, enhanced_prompt, selected_answer, cfg.max_new_tokens)
                pref["query_index"] = idx
                with open(pref_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(pref, ensure_ascii=False) + "\n")
                    pf.flush()
                    os.fsync(pf.fileno())

            # Clear preference collection variables
            del pref
        except Exception:
            mark_query_done(cfg.output_dir, state, idx)
            completed.add(idx)
            continue

        mark_query_done(cfg.output_dir, state, idx)
        completed.add(idx)

        # Clear response variables after preference collection
        del response_1, response_2, enhanced_prompt

        # This round's feedback was written after generation. Fit it before saving,
        # so a completed run includes every successful round in the checkpoint.
        if idx == len(training_queries) - 1:
            collected = PreferenceDataset(pref_path, include_critiques=use_critiques)
            if len(collected) > fitted:
                fit_preferences(collected, f"Final fit on {len(collected)} preferences")
            torch.save(generator.reward_model.state_dict(), final_ckpt)

    if not os.path.isfile(final_ckpt):
        collected = PreferenceDataset(pref_path, include_critiques=use_critiques)
        if len(collected) > 0:
            fit_preferences(collected, f"Final fit on {len(collected)} preferences")
        torch.save(generator.reward_model.state_dict(), final_ckpt)
    return final_ckpt

# ArmoRM-Llama3-8B is about 14 GiB in bfloat16. Leave headroom for the allocator.
ARMORM_LOAD_BYTES = 15 * 1024 ** 3
CPU_SCORE_BATCH_SIZE = 4

def scoring_device(requested_device: str, free_bytes: int) -> str:
    if str(requested_device).startswith("cuda") and free_bytes < ARMORM_LOAD_BYTES:
        return "cpu"
    return requested_device

def is_memory_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    return "out of memory" in text or "not enough free memory" in text

def _free_cuda_bytes(device: str) -> int:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    index = torch.cuda.current_device() if device == "cuda" else int(device.split(":", 1)[1])
    free, _total = torch.cuda.mem_get_info(index)
    return free

def _score_with_armorm(rows, attribute: str, device: str, batch_size: int):
    evaluator = None
    try:
        evaluator = PersonalizationRMEvaluator(
            attribute=attribute,
            device=device,
            rm_batch_size=batch_size,
        )
        (scored,) = evaluator.score_groups(rows)
        return scored
    finally:
        del evaluator
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

def score_responses(rows, attribute: str, device: str, batch_size: int):
    """Score on the GPU when it can hold ArmoRM; otherwise score on CPU."""
    notes = []
    gpu_device = device if scoring_device(device, _free_cuda_bytes(device) if str(device).startswith("cuda") else ARMORM_LOAD_BYTES) == device else None
    if str(device).startswith("cuda") and gpu_device is None:
        free = _free_cuda_bytes(device)
        notes.append(
            f"{device} has {free / 1024 ** 3:.2f} GiB free, below the "
            f"{ARMORM_LOAD_BYTES / 1024 ** 3:.0f} GiB needed to load ArmoRM."
        )
    if gpu_device is not None:
        try:
            scored = _score_with_armorm(rows, attribute, gpu_device, batch_size)
            return scored, gpu_device, batch_size, "\n".join(notes)
        except Exception as error:
            if not is_memory_error(error):
                raise
            notes.append(f"GPU scoring failed:\n{error}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    notes.append(f"Scored on CPU with batch size {CPU_SCORE_BATCH_SIZE}.")
    scored = _score_with_armorm(rows, attribute, "cpu", CPU_SCORE_BATCH_SIZE)
    return scored, "cpu", CPU_SCORE_BATCH_SIZE, "\n".join(notes)

def append_experiment_results(result_file: str, result: Dict, training_queries: int, evaluation_queries: int):
    with open(result_file, "a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 50 + "\n")
        handle.write("EXPERIMENT RESULTS\n")
        handle.write("=" * 50 + "\n")
        if result.get("error"):
            handle.write("ArmoRM scoring failed\n")
            handle.write(result["error"].rstrip() + "\n")
        else:
            handle.write(f"ArmoRM Score: {result['avg_reward']:.6f}\n")
        if "score_device" in result:
            handle.write(f"ArmoRM Device: {result['score_device']}\n")
            handle.write(f"ArmoRM Batch Size: {result['score_batch_size']}\n")
            if result.get("score_note"):
                handle.write(result["score_note"].rstrip() + "\n")
        handle.write(f"Total Responses: {result['total']}\n")
        handle.write(f"Training Queries: {training_queries}\n")
        handle.write(f"Evaluation Queries: {evaluation_queries}\n")
        if result.get("failed_generations", 0) > 0:
            handle.write(f"Failed Generations: {result['failed_generations']}\n")
        if "scores_file" in result:
            handle.write(f"ArmoRM Scored Responses: {result['scored_responses']}\n")
            handle.write(f"ArmoRM Score Failures: {result['score_failures']}\n")
            handle.write(f"ArmoRM Scores File: {result['scores_file']}\n")
        handle.write("=" * 50 + "\n")

def _record_eval_response(generator, question: Question, response: str, answers: Dict, responses_file: str) -> None:
    if not response or not response.strip():
        return

    response_data = {
        "question_id": question.question_id,
        "query": question.query,
        "ours": response
    }

    with open(responses_file, "a", encoding="utf-8") as rf:
        rf.write(json.dumps(response_data, ensure_ascii=False) + "\n")

    answers[question.question_id] = response
    total = len(answers)

    if hasattr(generator, 'last_topk_logits1'):
        generator.last_topk_logits1 = None
    if hasattr(generator, 'last_rewards1'):
        generator.last_rewards1 = None
    if hasattr(generator, 'last_topk_logits2'):
        generator.last_topk_logits2 = None
    if hasattr(generator, 'last_rewards2'):
        generator.last_rewards2 = None

    if total % 50 == 0:
        if hasattr(generator, "LLM") and hasattr(generator.LLM, 'past_key_values'):
            generator.LLM.past_key_values = None
        if hasattr(generator, "BLLM") and hasattr(generator.BLLM, 'past_key_values'):
            generator.BLLM.past_key_values = None
        torch.cuda.empty_cache()
    elif total % 10 == 0:
        torch.cuda.empty_cache()

def evaluate(generator: MABRewardGenerator, questions: List[Question], attribute: str,
             max_new_tokens: int, pre_screen_beam_width: int, cfg) -> Dict:

    answers = {}

    # Create directory for saving responses
    responses_dir = os.path.join(cfg.output_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    responses_file = os.path.join(responses_dir, f"responses_armo_rw_{cfg.reward_weight:.2f}.jsonl")
    answers.update(load_saved_responses(responses_file))
    pending = [question for question in questions if question.question_id not in answers]
    if answers:
        print(f"Resuming evaluation with {len(answers)} saved responses, {len(pending)} left.")

    # Generate responses for all queries. Parallel 1 keeps the original one-question decoder.
    parallel = configured_eval_parallel(cfg)
    print(f"Generating responses with eval_parallel={parallel}")
    with tqdm(total=len(pending), desc="Generating responses") as progress:
        for start in range(0, len(pending), parallel):
            chunk = pending[start:start + parallel]
            if parallel == 1:
                responses = [
                    generate_with_fixed_weight(
                        generator, question.query, max_new_tokens, pre_screen_beam_width, cfg.reward_weight,
                        nu_mab=cfg.nu_mab, lambda_mab=cfg.lambda_mab, attribute=attribute,
                    )
                    for question in chunk
                ]
            else:
                responses = generate_with_fixed_weight_batch(
                    generator, [question.query for question in chunk], max_new_tokens, pre_screen_beam_width,
                    cfg.reward_weight, nu_mab=cfg.nu_mab, lambda_mab=cfg.lambda_mab, attribute=attribute,
                )
            for question, response in zip(chunk, responses):
                progress.update(1)
                _record_eval_response(
                    generator, question, response, answers, responses_file,
                )

    total = len(answers)
    failed_generations = len(questions) - total
    print(f"\nEvaluating {total} responses with ArmoRM...")
    rows = make_score_rows(questions, answers)
    scores_file = os.path.join(cfg.output_dir, "scores.jsonl")
    if os.path.isfile(scores_file):
        os.remove(scores_file)
    summary = {
        "weight": cfg.reward_weight,
        "regularization": "weight_decay",
        "total": total,
        "failed_generations": failed_generations,
        "responses_file": responses_file,
        "evaluation_method": "ArmoRM",
        "scores_file": scores_file,
    }

    try:
        if hasattr(generator, 'LLM'):
            del generator.LLM
        if hasattr(generator, 'BLLM'):
            del generator.BLLM
        if hasattr(generator, 'bllm_tokenizer'):
            del generator.bllm_tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if total == 0:
            raise RuntimeError("No responses were generated.")
        batch_size = cfg.rm_batch_size if hasattr(cfg, 'rm_batch_size') else 32
        rows, score_device, score_batch_size, score_note = score_responses(
            rows, attribute, cfg.llm_device, batch_size
        )
        summary.update(score_device=score_device, score_batch_size=score_batch_size, score_note=score_note)
    except Exception:
        summary["error"] = traceback.format_exc()

    metrics = summarize_scores(rows)
    avg_reward = metrics["mean"]
    summary.update(avg_reward=avg_reward, scored_responses=metrics["scored"], score_failures=metrics["score_failures"])
    write_jsonl(scores_file, rows)
    if avg_reward is None:
        summary.setdefault("error", "ArmoRM did not return any finite scores.")
    if summary.get("error"):
        print(summary["error"])
    else:
        print(f"   Average ArmoRM reward: {avg_reward:.6f} ({score_device}, batch {score_batch_size})")
        if score_note:
            print(score_note)
    print(f"   Total responses: {total}")
    print(f"   Failed generations: {failed_generations}")
    print(f"   Responses saved to: {responses_file}")
    print(f"   Scores saved to: {scores_file}")
    return summary

def run(cfg):
    if not math.isfinite(cfg.critique_weight) or cfg.critique_weight < 0:
        raise ValueError("--critique_weight must be finite and non-negative")
    eval_parallel = configured_eval_parallel(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    result_file = os.path.join(cfg.output_dir, "result.txt")
    resuming = os.path.isfile(os.path.join(cfg.output_dir, RESUME_STATE))
    if not resuming and any(os.path.exists(os.path.join(cfg.output_dir, name)) for name in ("preferences", "checkpoints", "responses", "scores.jsonl")):
        raise ValueError("Use a fresh --output_dir for each run to avoid mixing feedback and checkpoints")
    if resuming and result_has_score(result_file):
        print(f"Already finished. Results: {result_file}")
        return
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 50)
    print(f"🧪 Starting Experiment (rw={cfg.reward_weight})")
    print("=" * 50)

    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Data file not found: {cfg.data_file}")

    questions = load_questions(cfg.data_file)
    all_queries = [question.query for question in questions]

    queries_path = os.path.join(cfg.output_dir, TRAINING_QUERIES)
    if os.path.isfile(queries_path):
        with open(queries_path, "r", encoding="utf-8") as handle:
            training_queries = json.load(handle)
    elif cfg.train_samples > len(all_queries):
        training_queries = all_queries.copy()
    else:
        training_queries = random.sample(all_queries, cfg.train_samples)
    if not os.path.isfile(queries_path):
        _atomic_json(queries_path, training_queries)
    if not resuming:
        save_resume_state(cfg.output_dir, {"completed_queries": [], "phase": "train"})

    settings_mode = "a" if resuming else "w"
    with open(result_file, settings_mode, encoding='utf-8') as sf:
        if resuming:
            sf.write(f"\nRESUME {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            sf.write("EXPERIMENT SETTINGS\n")
            sf.write("=" * 50 + "\n")
            sf.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sf.write(f"Data File: {cfg.data_file}\n")
            sf.write(f"LLM: {cfg.llm_path}\n")
            sf.write(f"Target attribute: {cfg.attribute}\n")
            sf.write(f"NU_MAB: {cfg.nu_mab}\n")
            sf.write(f"Pre-screen beam width: {cfg.pre_screen_beam_width}\n")
            sf.write(f"Max new tokens: {cfg.max_new_tokens}\n")
            sf.write(f"Regularization: Dynamic weight_decay = 1/(N + 50)\n")
            sf.write(f"Learning rate: {cfg.lr}\n")
            sf.write(f"Batch size: {cfg.batch_size}\n")
            sf.write(f"Epochs per query: {cfg.epochs_per_query}\n")
            sf.write(f"Normalize sentence embeddings: {cfg.normalize_sentence_embeddings}\n")
            sf.write(f"Reward Weight: {cfg.reward_weight}\n")
            sf.write(f"Training Samples: {cfg.train_samples}\n")
            sf.write(f"Random Seed: {cfg.seed}\n")
            sf.write(f"Eval parallel: {eval_parallel}\n")
            sf.write(f"Feedback mode: {cfg.feedback_mode}\n")
            sf.write(f"Critique weight: {cfg.critique_weight if cfg.feedback_mode == 'critique' else 0}\n")

    state = load_resume_state(cfg.output_dir)
    try:
        judge = PreferenceCollector(
            api_key=cfg.openai_api_key,
            model=cfg.preference_model_name,
            base_url=cfg.preference_api_base_url,
            sleep_time=cfg.preference_api_sleep_time,
            max_retries=cfg.preference_api_max_retries,
            collect_critiques=cfg.feedback_mode == "critique",
        )

        # Phase 1: Training
        print("Phase 1: Training reward model with dynamic weight_decay...")

        generator = MABRewardGenerator(
            llm_path=cfg.llm_path,
            reward_model_checkpoint_path=None,  # Start fresh
            embedding_model_path=cfg.embedding_model_path,
            llm_device=cfg.llm_device,
            reward_model_device=cfg.reward_model_device,
            embedding_model_device=cfg.embedding_model_device,
            torch_dtype=torch.float16,
            normalize_sentence_embeddings=cfg.normalize_sentence_embeddings,
        )

        if state["phase"] == "train":
            checkpoint_path = train_reward(
                generator=generator,
                judge=judge,
                attribute=cfg.attribute,
                training_queries=training_queries,
                cfg=cfg
            )
            state = load_resume_state(cfg.output_dir)
            state["phase"] = "eval"
            save_resume_state(cfg.output_dir, state)
        else:
            checkpoint_path = os.path.join(cfg.output_dir, "checkpoints", "final_model.pt")
            print("Phase 1 already finished. Skipping to evaluation.")

        # Phase 2: Evaluation
        print("Phase 2: Evaluating trained model...")

        if os.path.exists(checkpoint_path):
            generator.reward_model.load_state_dict(
                torch.load(checkpoint_path, map_location=cfg.reward_model_device)
            )
            pass

        result = evaluate(
            generator=generator,
            questions=questions,
            attribute=cfg.attribute,
            max_new_tokens=cfg.max_new_tokens,
            pre_screen_beam_width=cfg.pre_screen_beam_width,
            cfg=cfg
        )

        append_experiment_results(result_file, result, len(training_queries), len(questions))
        if result_has_score(result_file):
            state = load_resume_state(cfg.output_dir)
            state["phase"] = "done"
            save_resume_state(cfg.output_dir, state)

        if result.get("error"):
            print(f"ArmoRM scoring failed. The error is recorded in {result_file}")
        else:
            print(f"Completed: {result['avg_reward']:.6f} ArmoRM score ({result['total']} responses)")
        print(f"Responses: {result['responses_file']}")
        print(f"Results: {result_file}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Weight Decay Regularization Experiment")

    # Basic configuration
    parser.add_argument("--llm_path", required=True, 
                        help="Path to LLM model")
    parser.add_argument("--openai_api_key", default=os.getenv("TPOP2_API_KEY"),
                        required=not os.getenv("TPOP2_API_KEY"), help="API key or TPOP2_API_KEY")
    parser.add_argument("--attribute", default="creative",
                       help="Target attribute for alignment (creative, verbose, formal, etc.)")
    parser.add_argument("--data_file", required=True,
                       help="Path to JSON data file")
    parser.add_argument("--train_samples", type=int, required=True,
                       help="Number of samples to randomly select for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--embedding_model_path", default="Qwen/Qwen3-Embedding-0.6B",
                       help="Path to embedding model")
    parser.add_argument("--normalize_sentence_embeddings", action="store_true")
    parser.add_argument("--reward_weight", type=float, required=True,
                       help="Reward weight for generation (required)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--rm_batch_size", type=int, default=128,
                        help="ArmoRM evaluation batch size")
    parser.add_argument("--eval_parallel", type=int, default=1,
                        help="Number of evaluation questions answered at the same time")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--epochs_per_query", type=int, default=100,
                        help="Training epochs per query")

    parser.add_argument("--llm_device", default="cuda:1")
    parser.add_argument("--reward_model_device", default="cuda:0")
    parser.add_argument("--embedding_model_device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pre_screen_beam_width", type=int, default=40)
    parser.add_argument("--nu_mab", type=float, default=1.0,
                       help="Exploration parameter for MAB (nu in TTA algorithm)")
    parser.add_argument("--lambda_mab", type=float, default=1.0,
                       help="Regularization parameter for covariance matrix")

    parser.add_argument("--preference_model_name", required=True)
    parser.add_argument("--preference_api_base_url", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preference_api_sleep_time", type=float, default=1.0)
    parser.add_argument("--preference_api_max_retries", type=int, default=3)
    parser.add_argument("--feedback_mode", choices=("choice", "critique"), default="choice",
                        help="Use only A/B choices, or add an optional user evaluation after each choice")
    parser.add_argument("--critique_weight", type=float, default=0.5,
                        help="Weight of assistant revision comparisons in critique mode")

    cfg = parser.parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise
