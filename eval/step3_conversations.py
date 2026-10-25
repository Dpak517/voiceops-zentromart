"""STEP 3 - replay practice/conversations.csv turn by turn and score decisions, content and tool calls.

    python -m eval.step3_conversations --ids PC003,PC021        # a few conversations (cheap)
    python -m eval.step3_conversations --limit 3
    python -m eval.step3_conversations                          # all 23 conversations (~96 model calls!)
"""
import argparse
import os
import re
import time

import pandas as pd

from src import llm
from src.conversation import ConversationAgent
from src.scoring import check_exclude, check_include

_CALL = re.compile(r"(\w+)\((.*?)\)")


def parse_expected(spec):
    calls = []
    for name, args in _CALL.findall(str(spec or "")):
        kw = dict(re.findall(r"(\w+)=([^,\s]+)", args))
        calls.append((name, kw))
    return calls


def _digits(x):
    return re.sub(r"\D", "", str(x or ""))


def tools_ok(expected, actual):
    """Every expected call must appear (lookup_order: same order id and proof); no surprise tickets."""
    for name, kw in expected:
        found = False
        for a in actual:
            if a["tool"] != name:
                continue
            if name == "lookup_order":
                same_id = _digits(a["args"].get("order_id")) == _digits(kw.get("order_id"))
                same_proof = _digits(a["args"].get("verification_value")) == _digits(kw.get("verification_value"))
                found = found or (same_id and same_proof)
            else:
                found = True
        if not found:
            return False, f"missing {name}({kw})"
    if any(a["tool"] == "create_ticket" for a in actual) and not any(n == "create_ticket" for n, _ in expected):
        return False, "unexpected create_ticket"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="practice/conversations.csv")
    ap.add_argument("--ids", default=None, help="comma-separated conversation ids, e.g. PC003,PC021")
    ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/step3_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    cids = list(df.conversation_id.unique())
    if args.ids:
        cids = [c for c in cids if c in {x.strip() for x in args.ids.split(",")}]
    if args.limit:
        cids = cids[:args.limit]

    agent = ConversationAgent()
    rows = []
    for cid in cids:
        agent.reset()
        print(f"\n--- {cid} ({df[df.conversation_id == cid].scenario.iloc[0]}) ---")
        for r in df[df.conversation_id == cid].sort_values("turn", key=lambda s: s.astype(int)).to_dict("records"):
            try:
                out = agent.answer(r["customer_message"])
                err = ""
            except Exception as exc:                         # keep going; record the failure
                out = {"decision": "error", "reply": "", "tool_calls": [], "retrieved_ids": [], "state": {}}
                err = f"{type(exc).__name__}: {exc}"[:200]
            dec_ok = out["decision"] == r["expected_decision"]
            inc_ok, missing = check_include(out["reply"], r["must_include"])
            exc_ok, forbidden = check_exclude(out["reply"], r["must_not_include"])
            tl_ok, tl_why = tools_ok(parse_expected(r["expected_tool_calls"]), out["tool_calls"])
            passed = dec_ok and inc_ok and exc_ok and tl_ok and not err
            rows.append({
                "conversation_id": cid, "scenario": r["scenario"], "turn": r["turn"], "language": r["language"],
                "customer": r["customer_message"], "expected": r["expected_decision"], "got": out["decision"],
                "reply": out["reply"], "passed": passed, "decision_ok": dec_ok, "include_ok": inc_ok,
                "exclude_ok": exc_ok, "tools_ok": tl_ok, "missing": " | ".join(missing),
                "forbidden": " | ".join(forbidden), "tool_issue": tl_why, "error": err,
                "tools_called": " ; ".join(f"{t['tool']}({t['args']})" for t in out["tool_calls"]),
                "retrieved_ids": " ".join(out["retrieved_ids"]), "gold": r["gold_evidence"],
            })
            print(f"  t{r['turn']} {'PASS' if passed else 'FAIL'}  exp={r['expected_decision']:8} got={out['decision']:8}"
                  + (f"  [{'tools: ' + tl_why if not tl_ok else ''}{' missing: ' + ' | '.join(missing) if not inc_ok else ''}"
                     f"{' forbidden: ' + ' | '.join(forbidden) if not exc_ok else ''}{' ' + err if err else ''}]" if not passed else ""))
            time.sleep(args.sleep)

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False, encoding="utf-8")
    conv = res.groupby("conversation_id").passed.all()
    print("\n=== STEP 3 (conversations) ===")
    print(f"model: {llm.DEFAULT_MODEL}")
    print(f"Turns passed        : {res.passed.sum()}/{len(res)} = {res.passed.mean():.1%}")
    print(f"Conversations passed: {conv.sum()}/{len(conv)} (every turn right)")
    print(f"  decision ok {res.decision_ok.mean():.1%} | must_include ok {res.include_ok.mean():.1%} | "
          f"must_not ok {res.exclude_ok.mean():.1%} | tool calls ok {res.tools_ok.mean():.1%}")
    print("\nBy scenario (turns passed / total):")
    print(res.groupby("scenario").passed.agg(["sum", "count"]).to_string())
    print(f"\nSaved details to {args.out}")


if __name__ == "__main__":
    main()
