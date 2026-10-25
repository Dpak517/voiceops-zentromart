"""Scoring helpers for practice/questions.csv (reused unchanged in Step 2 and 3).

Conventions seen in the practice files:
  must_include      'A ; B / C'  -> need A  AND  (B OR C)       (';' = AND, ' / ' = OR)
  must_not_include  'X ; Y'      -> FAIL if X or Y appears anywhere in the reply
  acceptable_decisions 'answer ; escalate' -> any one of these is fine
"""
import re
import unicodedata

VALID_DECISIONS = {"answer", "clarify", "escalate", "refuse"}


def norm(text) -> str:
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", t).casefold().strip()


def and_groups(spec) -> list:
    return [g.strip() for g in re.split(r"\s*;\s*", str(spec or "")) if g.strip()]


def or_options(group) -> list:
    return [o.strip() for o in re.split(r"\s+/\s+", group) if o.strip()]


def check_include(reply, spec):
    """Return (ok, list_of_missing_groups)."""
    r = norm(reply)
    missing = [g for g in and_groups(spec)
               if not any(norm(o) in r for o in or_options(g))]
    return (not missing), missing


def check_exclude(reply, spec):
    """Return (ok, list_of_forbidden_phrases_found)."""
    r = norm(reply)
    bad = [o for g in and_groups(spec) for o in or_options(g) if norm(o) in r]
    return (not bad), bad


def acceptable_decisions(row) -> set:
    acc = set(and_groups(row.get("acceptable_decisions", "")))
    acc.add(str(row.get("expected_decision", "")).strip())
    return {a.casefold() for a in acc if a}


def score_question(row, decision, reply) -> dict:
    ok_dec = str(decision or "").strip().casefold() in acceptable_decisions(row)
    ok_inc, missing = check_include(reply, row.get("must_include", ""))
    ok_exc, forbidden = check_exclude(reply, row.get("must_not_include", ""))
    return {
        "decision_ok": ok_dec,
        "include_ok": ok_inc,
        "exclude_ok": ok_exc,
        "passed": ok_dec and ok_inc and ok_exc,
        "missing": " | ".join(missing),
        "forbidden_found": " | ".join(forbidden),
    }
