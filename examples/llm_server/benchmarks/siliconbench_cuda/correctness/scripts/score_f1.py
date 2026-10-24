#!/usr/bin/env python3
"""
Score responses against gold labels for the GMRID_v3 classification task.

Reads:
  responses.jsonl  — output of run_eval.py
  labels.jsonl     — output of build_prompts.py
  categories.json  — canonical 8-class taxonomy

Writes:
  scores.json — top-level F1 (weighted + macro), exact-match, per-class P/R/F1,
                parse-failure rate, and a small summary.

Metric choice mirrors inflaton/llms-at-edge calc_f1_score:
  f1_metric.compute(predictions, references, average="weighted")
with category names encoded to indices (OOV → -1), computed via sklearn.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)

from jsonl_io import iter_jsonl


METHOD_JSON = "json"
METHOD_REGEX = "regex_category"
METHOD_FUZZY = "fuzzy_name"
METHOD_NONE = "none"
METHOD_REQUEST_ERROR = "request_error"


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_CATEGORY_QUOTED_RE = re.compile(r'["\']category["\']\s*:\s*["\']([^"\']+)["\']')


def _extract_json_category(text: str) -> str | None:
    """raw_decode from the first '{' — accepts strict JSON or a single-quote variant."""
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    for candidate in (text[start:], text[start:].replace("'", '"')):
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "category" in obj:
            return str(obj["category"]).strip()
    return None


def extract_category(
    text: str,
    categories_lc_sorted: list[tuple[str, re.Pattern]],
) -> tuple[str | None, str]:
    """
    Pull a category string out of the model's text output.
    `categories_lc_sorted` is a list of (canonical_name, word-boundary regex)
    pre-sorted longest-first; built once per run.
    Returns (category_or_None, extraction_method).
    """
    if not text:
        return None, METHOD_NONE

    cleaned = text.strip()
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    cat = _extract_json_category(cleaned)
    if cat is not None:
        return cat, METHOD_JSON

    m = _CATEGORY_QUOTED_RE.search(text)
    if m:
        return m.group(1).strip(), METHOD_REGEX

    lowered = text.lower()
    for canonical, pattern in categories_lc_sorted:
        if pattern.search(lowered):
            return canonical, METHOD_FUZZY

    return None, METHOD_NONE


def normalize_to_canonical(predicted: str | None, categories: list[str]) -> str | None:
    """Case-insensitive match of predicted string to canonical category list."""
    if predicted is None:
        return None
    pred_l = predicted.strip().lower()
    for cat in categories:
        if cat.lower() == pred_l:
            return cat
    # Minor variations (e.g., "Weather Events" ↔ "Weather")
    for cat in categories:
        if cat.lower() in pred_l or pred_l in cat.lower():
            return cat
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True, help="responses.jsonl path")
    parser.add_argument("--labels", required=True, help="labels.jsonl path")
    parser.add_argument("--categories", default=None,
                        help="categories.json path (default: ../data/categories.json)")
    parser.add_argument("--output", required=True, help="scores.json path")
    parser.add_argument("--provenance", default=None,
                        help="JSON object describing which build produced these "
                             "responses (framework, version, model, machine, date). "
                             "Embedded verbatim under 'provenance' in scores.json. "
                             "Supplied by run_all_mac.sh so this script stays "
                             "independent of scripts/.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cats_path = (Path(args.categories) if args.categories
                 else script_dir.parent / "data" / "categories.json")

    with open(cats_path) as f:
        categories_json = json.load(f)
    categories = list(categories_json.keys())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Longest-first so "Worker Strike" is tried before any substring match
    categories_lc_sorted = [
        (c, re.compile(r"\b" + re.escape(c.lower()) + r"\b"))
        for c in sorted(categories, key=len, reverse=True)
    ]

    labels_by_id = {str(rec["id"]): rec["gold_category"]
                    for rec in iter_jsonl(args.labels)}
    responses_by_id = {str(rec["id"]): rec
                       for rec in iter_jsonl(args.responses)}

    lbl_keys, resp_keys = set(labels_by_id), set(responses_by_id)
    common_ids = sorted(lbl_keys & resp_keys)
    missing_responses = lbl_keys - resp_keys
    extra_responses = resp_keys - lbl_keys

    references: list[int] = []
    predictions: list[int] = []
    extraction_method_counts: Counter = Counter()
    request_errors = 0
    parse_failures = 0
    oov_predictions = 0
    per_id_detail: list[dict] = []

    for rid in common_ids:
        gold = labels_by_id[rid]
        resp = responses_by_id[rid]
        references.append(cat_to_idx[gold])

        if resp["error"]:
            request_errors += 1
            predictions.append(-1)
            extraction_method_counts[METHOD_REQUEST_ERROR] += 1
            per_id_detail.append({"id": rid, "gold": gold, "pred_raw": None,
                                  "pred_canonical": None,
                                  "method": METHOD_REQUEST_ERROR,
                                  "error": resp["error"]})
            continue

        text = (resp.get("content") or "") + (resp.get("reasoning") or "")
        raw_pred, method = extract_category(text, categories_lc_sorted)
        canonical = normalize_to_canonical(raw_pred, categories)
        extraction_method_counts[method] += 1

        if canonical is None:
            if method == METHOD_NONE:
                parse_failures += 1
            else:
                oov_predictions += 1

        predictions.append(cat_to_idx[canonical] if canonical is not None else -1)
        per_id_detail.append({"id": rid, "gold": gold, "pred_raw": raw_pred,
                              "pred_canonical": canonical, "method": method})

    n = len(common_ids)
    if n == 0:
        raise RuntimeError(
            f"No ids in common between {args.responses} and {args.labels}. "
            f"Did you run build_prompts and run_eval on matching splits?"
        )

    # sklearn F1 with -1 as an extra class == inflaton's encode-OOV-as-minus-one
    labels_for_metric = list(range(len(categories))) + [-1]
    f1_weighted = float(f1_score(references, predictions, labels=labels_for_metric,
                                 average="weighted", zero_division=0))
    f1_macro = float(f1_score(references, predictions, labels=labels_for_metric,
                              average="macro", zero_division=0))
    exact_match = float(accuracy_score(references, predictions))

    report = classification_report(
        references, predictions,
        labels=list(range(len(categories))),
        target_names=categories,
        output_dict=True, zero_division=0,
    )

    gold_counts = Counter(d["gold"] for d in per_id_detail)
    pred_counts = Counter(
        (d["pred_raw"] if d["pred_raw"] is not None else "<NO_PREDICTION>")
        for d in per_id_detail
    )

    summary = {
        "n_scored": n,
        "n_missing_responses": len(missing_responses),
        "n_extra_responses": len(extra_responses),
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "exact_match": exact_match,
        "request_errors": request_errors,
        "parse_failures": parse_failures,
        "oov_predictions_after_parse": oov_predictions,
        "extraction_methods": dict(extraction_method_counts),
        "gold_label_distribution": dict(gold_counts),
        "pred_label_distribution": dict(pred_counts),
        "per_class": report,
    }

    # Provenance is best-effort and never fatal: a malformed blob must not cost
    # us an eval that already ran. Record the parse failure instead of dropping
    # it silently, so a missing block always means "not supplied", never "lost".
    if args.provenance:
        try:
            summary["provenance"] = json.loads(args.provenance)
        except Exception as e:
            summary["provenance"] = {"error": f"unparseable provenance: {e}"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Scored: {n} (responses missing: {len(missing_responses)}, "
          f"extra: {len(extra_responses)})")
    print(f"  F1 weighted: {f1_weighted:.4f}")
    print(f"  F1 macro:    {f1_macro:.4f}")
    print(f"  Exact-match: {exact_match:.4f}")
    print(f"  Request errors:  {request_errors}")
    print(f"  Parse failures:  {parse_failures}")
    print(f"  OOV after parse: {oov_predictions}")
    print(f"  Extraction methods: {dict(extraction_method_counts)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
