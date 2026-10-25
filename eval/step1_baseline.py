"""STEP 1 - prompt-only baseline (no documents, no RAG).

Run from the repo root:
    python -m eval.step1_baseline                 # all 52 questions
    python -m eval.step1_baseline --limit 5       # quick smoke test
    python -m eval.step1_baseline --field asr_text   # simulate voice transcripts
"""
import argparse
import os
import time

import pandas as pd

from src import llm
from src.scoring import VALID_DECISIONS, score_question

SYSTEM_PROMPT = """You are ZentroMart's AI customer-support assistant. ZentroMart is an Indian online store selling groceries and electronics.

Rules:
- In your reply, briefly say you are ZentroMart's AI support assistant.
- Keep the reply to 2-3 short sentences.
- Reply in the language the customer used. If they write Hinglish (Hindi in Latin script), reply in Hinglish; if they write Hindi in Devanagari, reply in Hindi.
- Today's date is 20 October 2026.
- Always answer the customer's question directly and specifically (for example exact days or amounts), using your best knowledge of how Indian online stores usually work, even if you are not fully sure. Use "escalate" only if the customer asks for a human or reports fraud or a safety problem.
- Never promise compensation or exceptions.

Return ONLY a JSON object with exactly these keys:
  "decision": one of "answer" (you answered), "clarify" (you need more information), "escalate" (a human colleague must take over), "refuse" (the request breaks policy)
  "reply": the text to say to the customer
"""


def ask(question: str) -> tuple[str, str]:
    raw = llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question}],
        json_mode=True,
    )
    data = llm.parse_json(raw)
    decision = str(data.get("decision", "")).strip().lower()
    reply = str(data.get("reply", raw)).strip()
    return (decision if decision in VALID_DECISIONS else "invalid"), reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="practice/questions.csv")
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between calls (rate limits)")
    ap.add_argument("--out", default="outputs/step1_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for i, row in enumerate(df.to_dict("records"), 1):
        decision, reply = ask(row[args.field])
        s = score_question(row, decision, reply)
        rows.append({"id": row["id"], "intent": row["intent"], "language": row["language"],
                     "question": row[args.field], "expected": row["expected_decision"],
                     "got_decision": decision, "reply": reply, **s})
        print(f"[{i}/{len(df)}] {row['id']:8} {'PASS' if s['passed'] else 'FAIL'}  ({row['intent']})")
        time.sleep(args.sleep)

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False, encoding="utf-8")

    print("\n=== STEP 1 BASELINE ===")
    print(f"model: {llm.DEFAULT_MODEL} | input field: {args.field}")
    print(f"Overall pass: {res.passed.sum()}/{len(res)} = {res.passed.mean():.1%}")
    print(f"  decision ok: {res.decision_ok.mean():.1%} | must_include ok: {res.include_ok.mean():.1%}"
          f" | must_not_include ok: {res.exclude_ok.mean():.1%}")
    print("\nBy intent:\n", res.groupby("intent").passed.agg(["sum", "count"]).to_string())
    print("\nBy language:\n", res.groupby("language").passed.agg(["sum", "count"]).to_string())
    print(f"\nSaved details to {args.out}")


if __name__ == "__main__":
    main()
