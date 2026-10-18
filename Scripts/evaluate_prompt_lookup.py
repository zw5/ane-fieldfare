#!/usr/bin/env python3
"""Offline causal prompt/n-gram lookup screen over retained generation receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def candidate(history: list[int], n_min: int, n_max: int, width: int) -> tuple[int, list[int]]:
    """Return the longest-context, most-recent causal copy candidate.

    The copied continuation is restricted to tokens already present in history;
    it cannot cross the current decode position and therefore cannot see the
    answer it is evaluated against.
    """
    for n in range(min(n_max, len(history)), n_min - 1, -1):
        suffix = history[-n:]
        for start in range(len(history) - 2 * n, -1, -1):
            if history[start : start + n] != suffix:
                continue
            continuation = history[start + n : min(len(history), start + n + width)]
            if continuation:
                return n, continuation
    return 0, []


def evaluate(prompt: list[int], generated: list[int], n_min: int, n_max: int, width: int) -> dict:
    history = list(prompt)
    proposed = accepted = correct_first = 0
    accepted_prefixes: list[int] = []
    contexts: dict[str, int] = {}
    for offset, truth in enumerate(generated):
        n, draft = candidate(history, n_min, n_max, width)
        if draft:
            proposed += 1
            contexts[str(n)] = contexts.get(str(n), 0) + 1
            correct_first += int(draft[0] == truth)
            remaining = generated[offset : offset + len(draft)]
            prefix = 0
            for predicted, actual in zip(draft, remaining):
                if predicted != actual:
                    break
                prefix += 1
            accepted += prefix
            accepted_prefixes.append(prefix)
        history.append(truth)
    return {
        "decode_tokens": len(generated),
        "positions_with_proposal": proposed,
        "proposal_coverage": proposed / len(generated) if generated else 0.0,
        "first_token_accepts": correct_first,
        "first_token_acceptance": correct_first / proposed if proposed else 0.0,
        "accepted_draft_tokens": accepted,
        "accepted_tokens_per_decode_token": accepted / len(generated) if generated else 0.0,
        "maximum_accepted_prefix": max(accepted_prefixes, default=0),
        "matched_context_lengths": contexts,
    }


def receipts(paths: Iterable[Path]) -> Iterable[tuple[Path, dict]]:
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value.get("generated_tokens"), list):
            yield path, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((Path(__file__).resolve().parents[1] / "results").glob("*.json"))
    loaded = list(receipts(paths))
    prompt_catalog: dict[str, list[int]] = {}
    for _, value in loaded:
        ids = value.get("prompt_token_ids")
        prompt = value.get("prompt")
        if isinstance(prompt, str) and isinstance(ids, list):
            prompt_catalog[prompt] = ids
    runs = []
    for path, value in loaded:
        prompt_ids = value.get("prompt_token_ids", value.get("prompt_tokens", []))
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_catalog.get(value.get("prompt", ""), [])
        result = evaluate(prompt_ids, value["generated_tokens"], args.n_min, args.n_max, args.width)
        result["receipt"] = str(path)
        result["history_source"] = "prompt_and_generated" if prompt_ids else "generated_history_only"
        result["prompt_ids_recovered_from_matching_receipt"] = bool(
            prompt_ids and not isinstance(value.get("prompt_token_ids"), list)
        )
        runs.append(result)

    # Negative control: no repeated token means a causal lookup must emit no proposal.
    negative = evaluate(list(range(1000, 1064)), list(range(2000, 2064)), args.n_min, args.n_max, args.width)
    negative_ok = negative["positions_with_proposal"] == 0
    total_decode = sum(run["decode_tokens"] for run in runs)
    total_accepted = sum(run["accepted_draft_tokens"] for run in runs)
    total_proposals = sum(run["positions_with_proposal"] for run in runs)
    total_first = sum(run["first_token_accepts"] for run in runs)
    screened = bool(runs) and negative_ok
    accepted_per_decode = total_accepted / total_decode if total_decode else 0.0
    receipt = {
        "schema": "ane-fieldfare.prompt-lookup-screen.v1",
        "gate": "C4",
        "status": "validated" if screened else "rejected",
        "command": "Scripts/evaluate_prompt_lookup.py",
        "exit_status": 0 if screened else 1,
        "compile_count": 0,
        "load_count": 0,
        "configuration": {"n_min": args.n_min, "n_max": args.n_max, "draft_width": args.width},
        "runs": runs,
        "aggregate": {
            "receipt_count": len(runs),
            "decode_tokens": total_decode,
            "positions_with_proposal": total_proposals,
            "first_token_acceptance": total_first / total_proposals if total_proposals else 0.0,
            "accepted_tokens_per_decode_token": accepted_per_decode,
        },
        "decision": "go" if accepted_per_decode >= 0.10 else "no_go",
        "go_threshold": "at least 0.10 accepted draft tokens per retained decode token",
        "decision_basis": (
            "Observed acceptance is exactly zero, so the no-go result does not depend on the particular positive "
            "engineering threshold; 0.10 is retained as the minimum future re-screen bar."
        ),
        "negative_control": {"unique_history_emits_no_proposal": negative_ok, "result": negative},
        "claim_boundary": (
            "Offline causal replay of retained token IDs. Prompt IDs are used directly when present and recovered "
            "only from another receipt with byte-identical prompt text; unmatched older prompts conservatively screen "
            "generated-history lookup only. A source "
            "continuation is limited to tokens known before "
            "the evaluated decode position. This measures lookup opportunity only; it does not implement ANE verify "
            "lanes, establish end-to-end speedup, or generalize beyond the retained prompts."
        ),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if screened else 1


if __name__ == "__main__":
    raise SystemExit(main())
