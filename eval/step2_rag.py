"""STEP 2 - RAG evaluation on practice/questions.csv, with retrieval metrics and a Step 1 comparison.

Run from the repo root:
    python -m eval.step2_rag --limit 5
    python -m eval.step2_rag                       # all 52
    python -m eval.step2_rag --field asr_text      # simulated voice transcripts
    python -m eval.step2_rag --rewrite always      # ablation: LLM query rewrite for every question
"""
import argparse
import json
import os
import time

import pandas as pd

from src import llm
from src.rag_agent import RagAgent
from src.scoring import score_question


def gold_list(spec):
    return [g.strip() for g in str(spec).split(";") if g.strip()]


def recall(gold, found):
    return (sum(g in found for g in gold) / len(gold)) if gold else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="practice/questions.csv")
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated question ids to run, e.g. PQ0002,PQ0004")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rewrite", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--baseline", default="outputs/step1_guess.csv", help="Step 1 results to compare against")
    ap.add_argument("--out", default="outputs/step2_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if args.ids:
        df = df[df["id"].isin([x.strip() for x in args.ids.split(",")])]
    if args.limit:
        df = df.head(args.limit)
    agent = RagAgent(k=args.k, rewrite=args.rewrite)

    rows = []
    for i, row in enumerate(df.to_dict("records"), 1):
        out = agent.answer(row[args.field])
        s = score_question(row, out["decision"], out["reply"])
        gold = gold_list(row["gold_evidence"])
        rows.append({
            "id": row["id"], "intent": row["intent"], "language": row["language"],
            "question": row[args.field], "expected": row["expected_decision"],
            "got_decision": out["decision"], "reply": out["reply"], **s,
            "gold": " ; ".join(gold), "retrieved_ids": " ".join(out["retrieved_ids"]),
            "context_ids": " ".join(out["context_ids"]), "model_sources": " ".join(out["sources"]),
            "recall_search": recall(gold, set(out["retrieved_ids"])),
            "recall_context": recall(gold, set(out["context_ids"])),
            "search_query": out["search_query"], "rewrite": out["rewrite"],
            "tool_calls": json.dumps(out["tool_calls"], ensure_ascii=False),
        })
        print(f"[{i}/{len(df)}] {row['id']:8} {'PASS' if s['passed'] else 'FAIL'}  ({row['intent']})")
        time.sleep(args.sleep)

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False, encoding="utf-8")

    print("\n=== STEP 2 (RAG) ===")
    print(f"model: {llm.DEFAULT_MODEL} | k={args.k} | rewrite={args.rewrite} | input: {args.field}")
    print(f"Overall pass: {res.passed.sum()}/{len(res)} = {res.passed.mean():.1%}")
    print(f"  decision ok: {res.decision_ok.mean():.1%} | must_include ok: {res.include_ok.mean():.1%}"
          f" | must_not_include ok: {res.exclude_ok.mean():.1%}")
    has_gold = res[res.gold != ""]
    print(f"\nRetrieval (questions with gold evidence: {len(has_gold)})")
    print(f"  recall, search results only : {has_gold.recall_search.astype(float).mean():.1%}")
    print(f"  recall, incl. always-on docs: {has_gold.recall_context.astype(float).mean():.1%}")
    print("\nBy intent:\n", res.groupby("intent").passed.agg(["sum", "count"]).to_string())
    print("\nBy language:\n", res.groupby("language").passed.agg(["sum", "count"]).to_string())

    if os.path.exists(args.baseline):
        b = pd.read_csv(args.baseline, dtype=str, keep_default_na=False)
        b["passed"] = b["passed"].map({"True": True, "False": False})
        m = res.merge(b[["id", "passed"]], on="id", suffixes=("_step2", "_step1"))
        print(f"\n=== STEP 1 vs STEP 2 ({args.baseline}) ===")
        print(f"Step 1 (prompt only): {m.passed_step1.sum()}/{len(m)} = {m.passed_step1.mean():.1%}")
        print(f"Step 2 (RAG)        : {m.passed_step2.sum()}/{len(m)} = {m.passed_step2.mean():.1%}")
        cmp = m.groupby("intent")[["passed_step1", "passed_step2"]].sum().astype(int)
        print(cmp.to_string())
    print(f"\nSaved details to {args.out}  (open it in VS Code to read replies and see failures)")


if __name__ == "__main__":
    main()
