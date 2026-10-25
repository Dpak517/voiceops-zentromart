"""Sort failures from a results CSV by cause and write a readable report.

    python -m eval.show_failures                          # reads outputs/step2_results.csv
    python -m eval.show_failures --csv outputs/step2_results.csv
Open outputs/failures.txt in VS Code (it shows Hindi correctly; the terminal may not).
"""
import argparse

import pandas as pd


def cause(r):
    causes = []
    if str(r.get("gold", "")) and float(r.get("recall_context") or 0) < 1:
        causes.append("RETRIEVAL: a gold section was not in the prompt")
    if not (r["decision_ok"] == "True"):
        causes.append(f"DECISION: expected {r['expected']} but got {r['got_decision']}")
    if not (r["include_ok"] == "True"):
        causes.append("MISSING: " + str(r["missing"]))
    if not (r["exclude_ok"] == "True"):
        causes.append("FORBIDDEN: " + str(r["forbidden_found"]))
    return causes or ["unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/step2_results.csv")
    ap.add_argument("--out", default="outputs/failures.txt")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
    bad = df[df["passed"] != "True"]
    lines, tally = [], {}
    for _, r in bad.iterrows():
        cs = cause(r)
        for c in cs:
            key = c.split(":")[0]
            tally[key] = tally.get(key, 0) + 1
        lines += [f"=== {r['id']} | {r['intent']} | {r['language']} ===",
                  f"Q: {r['question']}",
                  f"REPLY ({r['got_decision']}): {r['reply']}",
                  "WHY: " + " | ".join(cs),
                  f"gold: {r.get('gold', '')}",
                  f"retrieved: {r.get('retrieved_ids', '')}",
                  f"tools: {str(r.get('tool_calls', ''))[:300]}", ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"{len(bad)} failures out of {len(df)}")
    print("Causes (a failure can have several):", tally)
    print(f"Details written to {a.out}")


if __name__ == "__main__":
    main()
